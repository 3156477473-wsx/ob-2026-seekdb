#ifndef OCEANBASE_SQL_RESOLVER_DML_OB_FILE_SQL_READER_H_
#define OCEANBASE_SQL_RESOLVER_DML_OB_FILE_SQL_READER_H_

#include <stdint.h>
#include <string>
#include <vector>


#include <glob.h>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

namespace oceanbase
{
namespace sql
{
namespace filesql
{

enum class FileType
{
  CSV = 0,
  JSONL,
  PARQUET,
};

enum class ColumnType
{
  NONE = 0,
  BOOLEAN,
  BIGINT,
  DOUBLE,
  DATE,
  DATETIME,
  VARCHAR,
};

struct Column
{
  std::string name;
  ColumnType type = ColumnType::NONE;
  bool nullable = false;
  int64_t length = 0;
};

using Schema = std::vector<Column>;

struct ReadOptions
{
  FileType type = FileType::CSV;
  bool has_header = true;
  int64_t sample_size = -1;
  char delimiter = ',';
  char quote = '"';
  std::string null_str;
};

int read_file_as_json(const std::string &path,
                      const ReadOptions &options,
                      Schema &schema,
                      std::string &json,
                      std::string &error);

int glob_files_as_json(const std::string &pattern, std::string &json, std::string &error);

const char *type_name(ColumnType type);

namespace
{

enum class Kind
{
  NULLV = 0,
  INT,
  DOUBLE,
  BOOL,
  DATE,
  DATETIME,
  STRING,
};

struct RawValue
{
  Kind kind = Kind::NULLV;
  std::string text;  // original text (normalized for dates)
};

void append_utf8(std::string &out, uint32_t cp)
{
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

bool is_space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

bool parse_date(const std::string &s)
{
  if (s.size() != 10) { return false; }
  for (int i = 0; i < 10; ++i) {
    if (i == 4 || i == 7) {
      if (s[i] != '-') { return false; }
    } else if (!isdigit(static_cast<unsigned char>(s[i]))) {
      return false;
    }
  }
  const int year = std::stoi(s.substr(0, 4));
  const int month = std::stoi(s.substr(5, 2));
  const int day = std::stoi(s.substr(8, 2));
  static const int mdays[] = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (month < 1 || month > 12 || day < 1) { return false; }
  int limit = mdays[month - 1];
  if (month == 2 && !(year % 4 == 0 && (year % 100 != 0 || year % 400 == 0))) { limit = 28; }
  return day <= limit;
}

bool parse_datetime(const std::string &s, std::string &normalized)
{
  if (s.size() != 19) { return false; }
  if (!parse_date(s.substr(0, 10))) { return false; }
  const char sep = s[10];
  if (sep != ' ' && sep != 'T') { return false; }
  if (s[13] != ':' || s[16] != ':') { return false; }
  for (int i = 11; i < 19; ++i) {
    if (i == 13 || i == 16) { continue; }
    if (!isdigit(static_cast<unsigned char>(s[i]))) { return false; }
  }
  const int hour = std::stoi(s.substr(11, 2));
  const int minute = std::stoi(s.substr(14, 2));
  const int second = std::stoi(s.substr(17, 2));
  if (hour > 23 || minute > 59 || second > 59) { return false; }
  normalized = s.substr(0, 10) + " " + s.substr(11);
  return true;
}

bool is_int_text(const std::string &s)
{
  if (s.empty()) { return false; }
  size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
  if (i >= s.size()) { return false; }
  for (; i < s.size(); ++i) {
    if (!isdigit(static_cast<unsigned char>(s[i]))) { return false; }
  }
  return true;
}

bool is_double_text(const std::string &s)
{
  if (s.empty()) { return false; }
  char *end = nullptr;
  errno = 0;
  const double v = std::strtod(s.c_str(), &end);
  (void)v;
  return errno == 0 && end != nullptr && *end == '\0' && end != s.c_str()
         && s.find_first_of(".eE") != std::string::npos;
}

bool is_bool_text(const std::string &s, bool &value)
{
  std::string lower;
  lower.reserve(s.size());
  for (char c : s) { lower.push_back(static_cast<char>(tolower(static_cast<unsigned char>(c)))); }
  if (lower == "true") { value = true; return true; }
  if (lower == "false") { value = false; return true; }
  return false;
}

RawValue classify_text(const std::string &text)
{
  RawValue v;
  if (text.empty()) {
    v.kind = Kind::NULLV;
    return v;
  }
  bool b = false;
  std::string normalized;
  if (is_bool_text(text, b)) {
    v.kind = Kind::BOOL;
    v.text = b ? "true" : "false";
  } else if (is_int_text(text)) {
    v.kind = Kind::INT;
    v.text = text;
  } else if (is_double_text(text)) {
    v.kind = Kind::DOUBLE;
    v.text = text;
  } else if (parse_datetime(text, normalized)) {
    v.kind = Kind::DATETIME;
    v.text = normalized;
  } else if (parse_date(text)) {
    v.kind = Kind::DATE;
    v.text = text;
  } else {
    v.kind = Kind::STRING;
    v.text = text;
  }
  return v;
}

ColumnType merge_type(ColumnType current, Kind kind)
{
  if (kind == Kind::NULLV) { return current; }
  ColumnType incoming = ColumnType::VARCHAR;
  switch (kind) {
    case Kind::BOOL: incoming = ColumnType::BOOLEAN; break;
    case Kind::INT: incoming = ColumnType::BIGINT; break;
    case Kind::DOUBLE: incoming = ColumnType::DOUBLE; break;
    case Kind::DATE: incoming = ColumnType::DATE; break;
    case Kind::DATETIME: incoming = ColumnType::DATETIME; break;
    default: incoming = ColumnType::VARCHAR; break;
  }
  if (current == ColumnType::NONE) { return incoming; }
  if (current == incoming) { return current; }
  const bool numeric = (current == ColumnType::BIGINT || current == ColumnType::DOUBLE)
                       && (incoming == ColumnType::BIGINT || incoming == ColumnType::DOUBLE);
  if (numeric) { return ColumnType::DOUBLE; }
  const bool temporal = (current == ColumnType::DATE || current == ColumnType::DATETIME)
                        && (incoming == ColumnType::DATE || incoming == ColumnType::DATETIME);
  if (temporal) { return ColumnType::DATETIME; }
  return ColumnType::VARCHAR;
}

void json_escape(const std::string &in, std::string &out)
{
  out.push_back('"');
  for (char c : in) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
          out += buf;
        } else {
          out.push_back(c);
        }
        break;
    }
  }
  out.push_back('"');
}

// Emits one value following the final column type so json_table sees a
// consistent representation.
void emit_value(const RawValue &raw, ColumnType type, std::string &out)
{
  if (raw.kind == Kind::NULLV) {
    out += "null";
    return;
  }
  if (type == ColumnType::BIGINT && raw.kind == Kind::INT) {
    out += raw.text;
    return;
  }
  if (type == ColumnType::DOUBLE && (raw.kind == Kind::INT || raw.kind == Kind::DOUBLE)) {
    out += raw.text;
    return;
  }
  if (type == ColumnType::BOOLEAN && raw.kind == Kind::BOOL) {
    out += raw.text;
    return;
  }
  json_escape(raw.text, out);
}

std::string dedup_name(const std::string &base,
                       const std::vector<std::string> &used,
                       int64_t index)
{
  std::string name = base;
  if (name.empty()) {
    name = "column" + std::to_string(index);
  }
  std::string candidate = name;
  int64_t suffix = 2;
  while (std::find(used.begin(), used.end(), candidate) != used.end()) {
    candidate = name + "__" + std::to_string(suffix);
    ++suffix;
  }
  return candidate;
}

int read_whole_file(const std::string &path, std::string &content, std::string &error)
{
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    error = "cannot open file: " + path;
    return -1;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  content = ss.str();
  return 0;
}

int parse_csv(const std::string &content,
              const ReadOptions &options,
              std::vector<std::vector<std::string>> &rows,
              std::string &error)
{
  std::vector<std::string> fields;
  std::string field;
  bool in_quote = false;
  bool row_started = false;
  const char delim = options.delimiter;
  const char quote = options.quote;
  for (size_t i = 0; i < content.size(); ++i) {
    const char c = content[i];
    if (in_quote) {
      if (c == quote) {
        if (i + 1 < content.size() && content[i + 1] == quote) {
          field.push_back(quote);
          ++i;
        } else {
          in_quote = false;
        }
      } else {
        field.push_back(c);
      }
    } else if (c == quote) {
      in_quote = true;
      row_started = true;
    } else if (c == delim) {
      fields.push_back(field);
      field.clear();
      row_started = true;
    } else if (c == '\n' || c == '\r') {
      if (c == '\r' && i + 1 < content.size() && content[i + 1] == '\n') { ++i; }
      if (row_started || !field.empty() || !fields.empty()) {
        fields.push_back(field);
        field.clear();
        rows.push_back(fields);
        fields.clear();
      }
      row_started = false;
    } else {
      field.push_back(c);
      row_started = true;
    }
  }
  if (in_quote) {
    error = "unterminated quoted field in CSV";
    return -1;
  }
  if (row_started || !field.empty() || !fields.empty()) {
    fields.push_back(field);
    rows.push_back(fields);
  }
  return 0;
}

int json_skip(const std::string &s, size_t &i)
{
  while (i < s.size() && is_space(s[i])) { ++i; }
  return 0;
}

int json_parse_string(const std::string &s, size_t &i, std::string &out)
{
  if (i >= s.size() || s[i] != '"') { return -1; }
  ++i;
  while (i < s.size()) {
    const char c = s[i++];
    if (c == '"') { return 0; }
    if (c != '\\') {
      out.push_back(c);
      continue;
    }
    if (i >= s.size()) { return -1; }
    const char esc = s[i++];
    switch (esc) {
      case '"': out.push_back('"'); break;
      case '\\': out.push_back('\\'); break;
      case '/': out.push_back('/'); break;
      case 'b': out.push_back('\b'); break;
      case 'f': out.push_back('\f'); break;
      case 'n': out.push_back('\n'); break;
      case 'r': out.push_back('\r'); break;
      case 't': out.push_back('\t'); break;
      case 'u': {
        if (i + 4 > s.size()) { return -1; }
        uint32_t cp = static_cast<uint32_t>(std::stoul(s.substr(i, 4), nullptr, 16));
        i += 4;
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
          const uint32_t low = static_cast<uint32_t>(std::stoul(s.substr(i + 2, 4), nullptr, 16));
          if (low >= 0xDC00 && low <= 0xDFFF) {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
            i += 6;
          }
        }
        append_utf8(out, cp);
        break;
      }
      default: return -1;
    }
  }
  return -1;
}

int json_parse_number(const std::string &s, size_t &i, std::string &out, bool &is_int)
{
  const size_t start = i;
  if (i < s.size() && (s[i] == '-' || s[i] == '+')) { ++i; }
  bool digits = false;
  bool dot = false;
  while (i < s.size()) {
    const char c = s[i];
    if (isdigit(static_cast<unsigned char>(c))) {
      digits = true;
      ++i;
    } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
      if (c == '.') { dot = true; }
      ++i;
    } else {
      break;
    }
  }
  if (!digits) { return -1; }
  out = s.substr(start, i - start);
  is_int = !dot && out.find_first_of("eE") == std::string::npos;
  return 0;
}

int json_parse_value(const std::string &s, size_t &i, RawValue &value)
{
  json_skip(s, i);
  if (i >= s.size()) { return -1; }
  const char c = s[i];
  if (c == '"') {
    std::string text;
    if (json_parse_string(s, i, text) != 0) { return -1; }
    value.kind = Kind::STRING;
    value.text = text;
    std::string normalized;
    if (text.size() == 10 && parse_date(text)) {
      value.kind = Kind::DATE;
    } else if (parse_datetime(text, normalized)) {
      value.kind = Kind::DATETIME;
      value.text = normalized;
    }
    return 0;
  }
  if (c == '{' || c == '[') {
    return -2;  // nested structure, caller reports a friendly error
  }
  if (s.compare(i, 4, "true") == 0) {
    value.kind = Kind::BOOL;
    value.text = "true";
    i += 4;
    return 0;
  }
  if (s.compare(i, 5, "false") == 0) {
    value.kind = Kind::BOOL;
    value.text = "false";
    i += 5;
    return 0;
  }
  if (s.compare(i, 4, "null") == 0) {
    value.kind = Kind::NULLV;
    value.text.clear();
    i += 4;
    return 0;
  }
  std::string number;
  bool is_int = false;
  if (json_parse_number(s, i, number, is_int) != 0) { return -1; }
  value.kind = is_int ? Kind::INT : Kind::DOUBLE;
  value.text = number;
  return 0;
}

}  // namespace

inline const char *type_name(ColumnType type)
{
  switch (type) {
    case ColumnType::BOOLEAN: return "BOOLEAN";
    case ColumnType::BIGINT: return "BIGINT";
    case ColumnType::DOUBLE: return "DOUBLE";
    case ColumnType::DATE: return "DATE";
    case ColumnType::DATETIME: return "DATETIME";
    case ColumnType::VARCHAR: return "VARCHAR";
    default: return "NULL";
  }
}

inline int read_file_as_json(const std::string &path,
                      const ReadOptions &options,
                      Schema &schema,
                      std::string &json,
                      std::string &error)
{
  std::string content;
  if (read_whole_file(path, content, error) != 0) { return -1; }
  if (options.sample_size != -1) {
    error = "sample_size other than -1 is not supported";
    return -1;
  }

  std::vector<std::string> names;
  std::vector<std::vector<RawValue>> rows;

  if (options.type == FileType::CSV) {
    std::vector<std::vector<std::string>> raw_rows;
    if (parse_csv(content, options, raw_rows, error) != 0) { return -1; }
    size_t first = 0;
    if (options.has_header && !raw_rows.empty()) {
      for (size_t c = 0; c < raw_rows[0].size(); ++c) {
        names.push_back(dedup_name(raw_rows[0][c], names, static_cast<int64_t>(c)));
      }
      first = 1;
    }
    for (size_t r = first; r < raw_rows.size(); ++r) {
      std::vector<RawValue> row;
      row.reserve(raw_rows[r].size());
      for (size_t c = 0; c < raw_rows[r].size(); ++c) {
        row.push_back(classify_text(raw_rows[r][c]));
      }
      rows.push_back(row);
    }
    if (names.empty()) {
      size_t width = 0;
      for (const auto &row : rows) { width = std::max(width, row.size()); }
      for (size_t c = 0; c < width; ++c) {
        names.push_back(dedup_name("", names, static_cast<int64_t>(c)));
      }
    }
  } else if (options.type == FileType::JSONL) {
    std::istringstream in(content);
    std::string line;
    std::map<std::string, size_t> key_index;
    int64_t line_no = 0;
    while (std::getline(in, line)) {
      ++line_no;
      if (!line.empty() && line.back() == '\r') { line.pop_back(); }
      if (line.find_first_not_of(" \t") == std::string::npos) { continue; }
      size_t i = 0;
      json_skip(line, i);
      if (i >= line.size() || line[i] != '{') {
        error = "line " + std::to_string(line_no) + ": JSONL record must be an object";
        return -1;
      }
      ++i;
      std::vector<std::pair<std::string, RawValue>> fields;
      json_skip(line, i);
      if (i < line.size() && line[i] == '}') {
        ++i;
      } else {
        while (true) {
          json_skip(line, i);
          std::string key;
          if (json_parse_string(line, i, key) != 0) {
            error = "line " + std::to_string(line_no) + ": invalid object key";
            return -1;
          }
          json_skip(line, i);
          if (i >= line.size() || line[i] != ':') {
            error = "line " + std::to_string(line_no) + ": expect ':'";
            return -1;
          }
          ++i;
          RawValue value;
          const int rc = json_parse_value(line, i, value);
          if (rc == -2) {
            error = "line " + std::to_string(line_no) + ": nested objects/arrays are not supported";
            return -1;
          } else if (rc != 0) {
            error = "line " + std::to_string(line_no) + ": invalid value";
            return -1;
          }
          fields.emplace_back(key, value);
          json_skip(line, i);
          if (i < line.size() && line[i] == ',') {
            ++i;
            continue;
          }
          if (i < line.size() && line[i] == '}') {
            ++i;
            break;
          }
          error = "line " + std::to_string(line_no) + ": expect ',' or '}'";
          return -1;
        }
      }
      json_skip(line, i);
      if (i != line.size()) {
        error = "line " + std::to_string(line_no) + ": trailing characters";
        return -1;
      }
      std::vector<RawValue> row;
      row.resize(names.size());
      for (const auto &field : fields) {
        auto it = key_index.find(field.first);
        if (it == key_index.end()) {
          const size_t index = names.size();
          key_index.emplace(field.first, index);
          names.push_back(field.first);
          row.resize(names.size());
          row[index] = field.second;
        } else {
          row[it->second] = field.second;
        }
      }
      if (row.size() < names.size()) { row.resize(names.size()); }
      rows.push_back(row);
    }
  } else {
    error = "parquet is not supported yet";
    return -1;
  }

  schema.clear();
  schema.resize(names.size());
  for (size_t c = 0; c < names.size(); ++c) {
    schema[c].name = names[c];
    schema[c].type = ColumnType::NONE;
  }
  for (const auto &row : rows) {
    for (size_t c = 0; c < names.size() && c < row.size(); ++c) {
      if (row[c].kind == Kind::NULLV) {
        schema[c].nullable = true;
      } else {
        schema[c].type = merge_type(schema[c].type, row[c].kind);
      }
    }
  }
  for (size_t c = 0; c < schema.size(); ++c) {
    if (schema[c].type == ColumnType::NONE) {
      schema[c].type = ColumnType::VARCHAR;
    }
    int64_t length = 0;
    for (const auto &row : rows) {
      if (c < row.size() && row[c].kind != Kind::NULLV) {
        length = std::max<int64_t>(length, static_cast<int64_t>(row[c].text.size()));
      }
    }
    schema[c].length = std::max<int64_t>(length, 1);
  }

  json.clear();
  json.reserve(rows.size() * 64 + 16);
  json.push_back('[');
  for (size_t r = 0; r < rows.size(); ++r) {
    if (r > 0) { json.push_back(','); }
    json.push_back('{');
    for (size_t c = 0; c < schema.size(); ++c) {
      if (c > 0) { json.push_back(','); }
      json_escape(schema[c].name, json);
      json.push_back(':');
      static const RawValue kNull;
      const RawValue &value = (c < rows[r].size()) ? rows[r][c] : kNull;
      emit_value(value, schema[c].type, json);
    }
    json.push_back('}');
  }
  json.push_back(']');
  return 0;
}

inline int glob_files_as_json(const std::string &pattern, std::string &json, std::string &error)
{
  glob_t matches;
  memset(&matches, 0, sizeof(matches));
  const int rc = ::glob(pattern.c_str(), GLOB_NOSORT, nullptr, &matches);
  if (rc != 0 && rc != GLOB_NOMATCH) {
    error = "invalid glob pattern: " + pattern;
    ::globfree(&matches);
    return -1;
  }
  std::vector<std::string> files;
  for (size_t i = 0; i < matches.gl_pathc; ++i) {
    files.emplace_back(matches.gl_pathv[i]);
  }
  ::globfree(&matches);
  std::sort(files.begin(), files.end());
  json.clear();
  json.push_back('[');
  for (size_t i = 0; i < files.size(); ++i) {
    if (i > 0) { json.push_back(','); }
    json += "{\"file\":";
    json_escape(files[i], json);
    json.push_back('}');
  }
  json.push_back(']');
  return 0;
}

// ---------------------------------------------------------------------------
// Minimal Parquet reader (flat schemas, PLAIN / RLE_DICTIONARY, UNCOMPRESSED /
// SNAPPY).  Supports DataPageV1, DataPageV2 and DictionaryPage.
// ---------------------------------------------------------------------------
namespace pq
{

struct SchemaElem
{
  int32_t type = -1;
  int32_t repetition = -1;
  std::string name;
  int32_t num_children = 0;
};

struct ColMeta
{
  int32_t type = -1;
  int32_t codec = 0;
  int64_t num_values = 0;
  int64_t data_page_offset = 0;
  int64_t dictionary_page_offset = 0;
  int64_t total_compressed_size = 0;
};

struct FileMeta
{
  std::vector<SchemaElem> schema;
  std::vector<std::vector<ColMeta>> row_groups;
  int64_t num_rows = 0;
};

inline uint32_t read_le32(const unsigned char *p)
{
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline uint64_t read_le64(const unsigned char *p)
{
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) { v = (v << 8) | p[i]; }
  return v;
}

class CompactReader
{
public:
  CompactReader(const unsigned char *data, size_t size) : data_(data), size_(size) {}

  bool ok() const { return pos_ <= size_; }

  int read_byte(unsigned char &v)
  {
    if (pos_ >= size_) { return -1; }
    v = data_[pos_++];
    return 0;
  }

  int read_varint(uint64_t &v)
  {
    v = 0;
    int shift = 0;
    while (true) {
      unsigned char b = 0;
      if (read_byte(b) != 0) { return -1; }
      v |= (static_cast<uint64_t>(b & 0x7F) << shift);
      if ((b & 0x80) == 0) { break; }
      shift += 7;
      if (shift > 63) { return -1; }
    }
    return 0;
  }

  int read_zigzag(int64_t &v)
  {
    uint64_t u = 0;
    if (read_varint(u) != 0) { return -1; }
    v = static_cast<int64_t>((u >> 1) ^ (~(u & 1) + 1));
    return 0;
  }

  int read_i32(int32_t &v)
  {
    int64_t z = 0;
    if (read_zigzag(z) != 0) { return -1; }
    v = static_cast<int32_t>(z);
    return 0;
  }

  int read_i64(int64_t &v) { return read_zigzag(v); }

  int read_binary(std::string &v)
  {
    uint64_t len = 0;
    if (read_varint(len) != 0) { return -1; }
    if (pos_ + len > size_) { return -1; }
    v.assign(reinterpret_cast<const char *>(data_ + pos_), static_cast<size_t>(len));
    pos_ += static_cast<size_t>(len);
    return 0;
  }

  int read_struct_header(bool &stop, int &type, int32_t &field_id)
  {
    stop = false;
    unsigned char b = 0;
    if (read_byte(b) != 0) { return -1; }
    if (b == 0) {
      stop = true;
      return 0;
    }
    type = b & 0x0F;
    const int32_t delta = (b >> 4) & 0x0F;
    if (delta != 0) {
      field_id += delta;
    } else {
      int64_t z = 0;
      if (read_zigzag(z) != 0) { return -1; }
      field_id = static_cast<int32_t>(z);
    }
    return 0;
  }

  int skip(int type)
  {
    switch (type) {
      case 1:   // bool true
      case 2:   // bool false
        return 0;
      case 3: {  // byte
        unsigned char b = 0;
        return read_byte(b);
      }
      case 4:
      case 5:
      case 6: {
        uint64_t u = 0;
        return read_varint(u) == 0 ? 0 : -1;
      }
      case 7: {  // double
        if (pos_ + 8 > size_) { return -1; }
        pos_ += 8;
        return 0;
      }
      case 8: {
        std::string tmp;
        return read_binary(tmp);
      }
      case 9:
      case 10: {  // list / set
        unsigned char b = 0;
        if (read_byte(b) != 0) { return -1; }
        uint64_t n = (b >> 4) & 0x0F;
        const int elem_type = b & 0x0F;
        if (n == 15) {
          if (read_varint(n) != 0) { return -1; }
        }
        for (uint64_t i = 0; i < n; ++i) {
          if (skip(elem_type) != 0) { return -1; }
        }
        return 0;
      }
      case 11: {  // map
        uint64_t n = 0;
        if (read_varint(n) != 0) { return -1; }
        if (n > 0) {
          unsigned char b = 0;
          if (read_byte(b) != 0) { return -1; }
          const int kt = (b >> 4) & 0x0F;
          const int vt = b & 0x0F;
          for (uint64_t i = 0; i < n; ++i) {
            if (skip(kt) != 0 || skip(vt) != 0) { return -1; }
          }
        }
        return 0;
      }
      case 12: {  // struct
        int32_t fid = 0;
        while (true) {
          bool stop = false;
          int t = 0;
          if (read_struct_header(stop, t, fid) != 0) { return -1; }
          if (stop) { break; }
          if (skip(t) != 0) { return -1; }
        }
        return 0;
      }
      default:
        return -1;
    }
  }

  int next_list(unsigned char &b, uint64_t &n, int &elem_type)
  {
    if (read_byte(b) != 0) { return -1; }
    n = (b >> 4) & 0x0F;
    elem_type = b & 0x0F;
    if (n == 15) {
      if (read_varint(n) != 0) { return -1; }
    }
    return 0;
  }

  size_t pos() const { return pos_; }
  void seek(size_t pos) { pos_ = pos; }

private:
  const unsigned char *data_;
  size_t size_;
  size_t pos_ = 0;
};

inline int parse_schema_elem(CompactReader &r, SchemaElem &elem)
{
  int32_t fid = 0;
  while (true) {
    bool stop = false;
    int type = 0;
    if (r.read_struct_header(stop, type, fid) != 0) { return -1; }
    if (stop) { break; }
    if (fid == 1 && type == 5) {
      if (r.read_i32(elem.type) != 0) { return -1; }
    } else if (fid == 3 && type == 5) {
      if (r.read_i32(elem.repetition) != 0) { return -1; }
    } else if (fid == 4 && type == 8) {
      if (r.read_binary(elem.name) != 0) { return -1; }
    } else if (fid == 5 && type == 5) {
      if (r.read_i32(elem.num_children) != 0) { return -1; }
    } else if (r.skip(type) != 0) {
      return -1;
    }
  }
  return 0;
}

inline int parse_col_meta(CompactReader &r, ColMeta &meta)
{
  int32_t fid = 0;
  while (true) {
    bool stop = false;
    int type = 0;
    if (r.read_struct_header(stop, type, fid) != 0) { return -1; }
    if (stop) { break; }
    if (fid == 1 && type == 5) {
      if (r.read_i32(meta.type) != 0) { return -1; }
    } else if (fid == 4 && type == 5) {
      if (r.read_i32(meta.codec) != 0) { return -1; }
    } else if (fid == 5 && type == 6) {
      if (r.read_i64(meta.num_values) != 0) { return -1; }
    } else if (fid == 7 && type == 6) {
      if (r.read_i64(meta.total_compressed_size) != 0) { return -1; }
    } else if (fid == 9 && type == 6) {
      if (r.read_i64(meta.data_page_offset) != 0) { return -1; }
    } else if (fid == 11 && type == 6) {
      if (r.read_i64(meta.dictionary_page_offset) != 0) { return -1; }
    } else if (r.skip(type) != 0) {
      return -1;
    }
  }
  return 0;
}

inline int parse_column_chunk(CompactReader &r, ColMeta &meta)
{
  int32_t fid = 0;
  while (true) {
    bool stop = false;
    int type = 0;
    if (r.read_struct_header(stop, type, fid) != 0) { return -1; }
    if (stop) { break; }
    if (fid == 3 && type == 12) {
      if (parse_col_meta(r, meta) != 0) { return -1; }
    } else if (r.skip(type) != 0) {
      return -1;
    }
  }
  return 0;
}

inline int parse_row_group(CompactReader &r, std::vector<ColMeta> &cols)
{
  int32_t fid = 0;
  while (true) {
    bool stop = false;
    int type = 0;
    if (r.read_struct_header(stop, type, fid) != 0) { return -1; }
    if (stop) { break; }
    if (fid == 1 && type == 9) {
      unsigned char b = 0;
      uint64_t n = 0;
      int et = 0;
      if (r.next_list(b, n, et) != 0) { return -1; }
      for (uint64_t i = 0; i < n; ++i) {
        ColMeta meta;
        if (parse_column_chunk(r, meta) != 0) { return -1; }
        cols.push_back(meta);
      }
    } else if (r.skip(type) != 0) {
      return -1;
    }
  }
  return 0;
}

inline int parse_file_meta(const unsigned char *data, size_t size, FileMeta &meta)
{
  CompactReader r(data, size);
  int32_t fid = 0;
  while (true) {
    bool stop = false;
    int type = 0;
    if (r.read_struct_header(stop, type, fid) != 0) { return -1; }
    if (stop) { break; }
    if (fid == 2 && type == 9) {
      unsigned char b = 0;
      uint64_t n = 0;
      int et = 0;
      if (r.next_list(b, n, et) != 0) { return -1; }
      for (uint64_t i = 0; i < n; ++i) {
        SchemaElem elem;
        if (parse_schema_elem(r, elem) != 0) { return -1; }
        meta.schema.push_back(elem);
      }
    } else if (fid == 3 && type == 6) {
      if (r.read_i64(meta.num_rows) != 0) { return -1; }
    } else if (fid == 4 && type == 9) {
      unsigned char b = 0;
      uint64_t n = 0;
      int et = 0;
      if (r.next_list(b, n, et) != 0) { return -1; }
      for (uint64_t i = 0; i < n; ++i) {
        std::vector<ColMeta> cols;
        if (parse_row_group(r, cols) != 0) { return -1; }
        meta.row_groups.push_back(cols);
      }
    } else if (r.skip(type) != 0) {
      return -1;
    }
  }
  return 0;
}

inline int snappy_uncompress(const unsigned char *in, size_t in_size,
                             std::string &out, size_t expected)
{
  out.clear();
  out.reserve(expected);
  size_t i = 0;
  uint64_t total = 0;
  int shift = 0;
  while (i < in_size) {
    const unsigned char b = in[i++];
    total |= (static_cast<uint64_t>(b & 0x7F) << shift);
    if ((b & 0x80) == 0) { break; }
    shift += 7;
    if (shift > 35) { return -1; }
  }
  while (i < in_size) {
    const unsigned char tag = in[i++];
    const int kind = tag & 0x03;
    if (kind == 0) {
      size_t len = (tag >> 2) + 1;
      if (len > 60) {
        const size_t extra = len - 60;
        if (i + extra > in_size) { return -1; }
        len = 0;
        for (size_t k = 0; k < extra; ++k) {
          len |= static_cast<size_t>(in[i + k]) << (8 * k);
        }
        len += 1;
        i += extra;
      }
      if (i + len > in_size) { return -1; }
      out.append(reinterpret_cast<const char *>(in + i), len);
      i += len;
    } else {
      size_t len = 0;
      size_t offset = 0;
      if (kind == 1) {
        if (i >= in_size) { return -1; }
        len = 4 + ((tag >> 2) & 0x07);
        offset = (static_cast<size_t>(tag >> 5) << 8) | in[i++];
      } else if (kind == 2) {
        if (i + 2 > in_size) { return -1; }
        len = (tag >> 2) + 1;
        offset = static_cast<size_t>(in[i]) | (static_cast<size_t>(in[i + 1]) << 8);
        i += 2;
      } else {
        if (i + 4 > in_size) { return -1; }
        len = (tag >> 2) + 1;
        offset = static_cast<size_t>(in[i]) | (static_cast<size_t>(in[i + 1]) << 8)
                 | (static_cast<size_t>(in[i + 2]) << 16) | (static_cast<size_t>(in[i + 3]) << 24);
        i += 4;
      }
      if (offset == 0 || offset > out.size()) { return -1; }
      const size_t start = out.size() - offset;
      for (size_t k = 0; k < len; ++k) {
        out.push_back(out[start + (k % offset)]);
      }
    }
  }
  if (total != 0 && out.size() != static_cast<size_t>(total)) { return -1; }
  return 0;
}

inline uint32_t read_packed_bits(const unsigned char *p, size_t &bit_pos, int width)
{
  uint32_t v = 0;
  for (int i = 0; i < width; ++i) {
    const size_t byte = bit_pos >> 3;
    const int bit = static_cast<int>(bit_pos & 7);
    v |= (static_cast<uint32_t>((p[byte] >> bit) & 1) << i);
    ++bit_pos;
  }
  return v;
}

// RLE / bit-packed hybrid (parquet spec), without the 4 byte length prefix.
inline int rle_hybrid_decode(const unsigned char *data, size_t size, int width,
                             size_t count, std::vector<uint32_t> &out)
{
  out.clear();
  out.reserve(count);
  size_t pos = 0;
  while (out.size() < count && pos < size) {
    uint64_t header = 0;
    int shift = 0;
    while (true) {
      if (pos >= size) { return 0; }
      const unsigned char b = data[pos++];
      header |= (static_cast<uint64_t>(b & 0x7F) << shift);
      if ((b & 0x80) == 0) { break; }
      shift += 7;
      if (shift > 35) { return -1; }
    }
    if ((header & 1) == 0) {
      const size_t run = static_cast<size_t>(header >> 1);
      const size_t nbytes = (static_cast<size_t>(width) + 7) / 8;
      if (pos + nbytes > size) { return -1; }
      uint32_t value = 0;
      for (size_t k = 0; k < nbytes; ++k) {
        value |= (static_cast<uint32_t>(data[pos + k]) << (8 * k));
      }
      pos += nbytes;
      for (size_t k = 0; k < run && out.size() < count; ++k) { out.push_back(value); }
    } else {
      const size_t groups = static_cast<size_t>(header >> 1);
      const size_t nbytes = groups * static_cast<size_t>(width);
      if (pos + nbytes > size) { return -1; }
      size_t bit_pos = pos * 8;
      const size_t total = groups * 8;
      for (size_t k = 0; k < total && out.size() < count; ++k) {
        out.push_back(read_packed_bits(data, bit_pos, width));
      }
      pos += nbytes;
    }
  }
  return 0;
}

inline int plain_decode_values(const unsigned char *data, size_t size, int32_t type,
                               size_t count, std::vector<std::string> &values)
{
  size_t pos = 0;
  for (size_t i = 0; i < count; ++i) {
    if (type == 0) {  // BOOLEAN
      const size_t byte = i >> 3;
      if (byte >= size) { return -1; }
      values.emplace_back(((data[byte] >> (i & 7)) & 1) ? "true" : "false");
    } else if (type == 1) {  // INT32
      if (pos + 4 > size) { return -1; }
      values.emplace_back(std::to_string(static_cast<int32_t>(read_le32(data + pos))));
      pos += 4;
    } else if (type == 2) {  // INT64
      if (pos + 8 > size) { return -1; }
      values.emplace_back(std::to_string(static_cast<int64_t>(read_le64(data + pos))));
      pos += 8;
    } else if (type == 4) {  // FLOAT
      if (pos + 4 > size) { return -1; }
      const uint32_t bits = read_le32(data + pos);
      float f = 0;
      memcpy(&f, &bits, sizeof(f));
      char buf[64];
      snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(f));
      values.emplace_back(buf);
      pos += 4;
    } else if (type == 5) {  // DOUBLE
      if (pos + 8 > size) { return -1; }
      const uint64_t bits = read_le64(data + pos);
      double d = 0;
      memcpy(&d, &bits, sizeof(d));
      char buf[64];
      snprintf(buf, sizeof(buf), "%.17g", d);
      values.emplace_back(buf);
      pos += 8;
    } else if (type == 6) {  // BYTE_ARRAY
      if (pos + 4 > size) { return -1; }
      const uint32_t len = read_le32(data + pos);
      pos += 4;
      if (pos + len > size) { return -1; }
      values.emplace_back(reinterpret_cast<const char *>(data + pos), len);
      pos += len;
    } else {
      return -1;
    }
  }
  return 0;
}

struct ColumnData
{
  std::vector<std::string> values;   // textual representation per present value
  std::vector<uint8_t> present;      // 0 = null
};

inline int decode_column(const std::string &file, const ColMeta &meta,
                         bool optional, ColumnData &out, std::string &error)
{
  const unsigned char *base = reinterpret_cast<const unsigned char *>(file.data());
  int64_t offset = meta.dictionary_page_offset > 0 ? meta.dictionary_page_offset
                                                   : meta.data_page_offset;
  int64_t end = offset + meta.total_compressed_size;
  if (offset < 4 || end > static_cast<int64_t>(file.size())) {
    error = "parquet column offset out of range";
    return -1;
  }
  std::vector<std::string> dictionary;
  while (offset < end) {
    CompactReader r(base + offset, static_cast<size_t>(end - offset));
    int32_t page_type = -1;
    int32_t uncompressed = 0;
    int32_t compressed = 0;
    int32_t num_values = 0;
    int32_t encoding = -1;
    int32_t def_encoding = -1;
    int32_t rep_encoding = -1;
    int32_t dict_num_values = 0;
    int32_t v2_num_values = 0;
    int32_t v2_def_len = 0;
    int32_t v2_rep_len = 0;
    bool v2_compressed = true;
    int32_t fid = 0;
    while (true) {
      bool stop = false;
      int type = 0;
      if (r.read_struct_header(stop, type, fid) != 0) {
        error = "bad page header";
        return -1;
      }
      if (stop) { break; }
      if (fid == 1 && type == 5) {
        if (r.read_i32(page_type) != 0) { return -1; }
      } else if (fid == 2 && type == 5) {
        if (r.read_i32(uncompressed) != 0) { return -1; }
      } else if (fid == 3 && type == 5) {
        if (r.read_i32(compressed) != 0) { return -1; }
      } else if (fid == 5 && type == 12) {  // data_page_header
        int32_t hfid = 0;
        while (true) {
          bool hstop = false;
          int htype = 0;
          if (r.read_struct_header(hstop, htype, hfid) != 0) { return -1; }
          if (hstop) { break; }
          if (hfid == 1 && htype == 5) {
            if (r.read_i32(num_values) != 0) { return -1; }
          } else if (hfid == 2 && htype == 5) {
            if (r.read_i32(encoding) != 0) { return -1; }
          } else if (hfid == 3 && htype == 5) {
            if (r.read_i32(def_encoding) != 0) { return -1; }
          } else if (hfid == 4 && htype == 5) {
            if (r.read_i32(rep_encoding) != 0) { return -1; }
          } else if (r.skip(htype) != 0) { return -1; }
        }
      } else if (fid == 7 && type == 12) {  // dictionary_page_header
        int32_t hfid = 0;
        while (true) {
          bool hstop = false;
          int htype = 0;
          if (r.read_struct_header(hstop, htype, hfid) != 0) { return -1; }
          if (hstop) { break; }
          if (hfid == 1 && htype == 5) {
            if (r.read_i32(dict_num_values) != 0) { return -1; }
          } else if (r.skip(htype) != 0) { return -1; }
        }
      } else if (fid == 8 && type == 12) {  // data_page_header_v2
        int32_t hfid = 0;
        while (true) {
          bool hstop = false;
          int htype = 0;
          if (r.read_struct_header(hstop, htype, hfid) != 0) { return -1; }
          if (hstop) { break; }
          if (hfid == 1 && htype == 5) {
            if (r.read_i32(v2_num_values) != 0) { return -1; }
          } else if (hfid == 5 && htype == 5) {
            if (r.read_i32(v2_def_len) != 0) { return -1; }
          } else if (hfid == 6 && htype == 5) {
            if (r.read_i32(v2_rep_len) != 0) { return -1; }
          } else if (hfid == 7 && htype == 1) {
            v2_compressed = true;
          } else if (hfid == 7 && htype == 2) {
            v2_compressed = false;
          } else if (hfid == 4 && htype == 5) {
            if (r.read_i32(encoding) != 0) { return -1; }
          } else if (r.skip(htype) != 0) { return -1; }
        }
      } else if (r.skip(type) != 0) {
        return -1;
      }
    }
    const size_t header_len = r.pos();
    const unsigned char *body = base + offset + header_len;
    if (body + compressed > base + end) {
      error = "truncated parquet page";
      return -1;
    }
    std::string page;
    if (3 == page_type) {
      // DataPageV2: repetition/definition levels stay uncompressed and only the
      // value section is compressed, so slice the payload before decompressing.
      const size_t prefix = static_cast<size_t>(v2_rep_len) + static_cast<size_t>(v2_def_len);
      if (prefix > static_cast<size_t>(compressed) || prefix > static_cast<size_t>(uncompressed)) {
        error = "bad v2 page sizes";
        return -1;
      }
      page.assign(reinterpret_cast<const char *>(body), prefix);
      const unsigned char *valbody = body + prefix;
      const size_t valbytes = static_cast<size_t>(compressed) - prefix;
      const size_t valexpect = static_cast<size_t>(uncompressed) - prefix;
      if (!v2_compressed) {
        page.append(reinterpret_cast<const char *>(valbody), valbytes);
      } else if (meta.codec == 1) {
        std::string vals;
        if (snappy_uncompress(valbody, valbytes, vals, valexpect) != 0) {
          error = "snappy decompress failed (v2) c=" + std::to_string(compressed)
                  + " u=" + std::to_string(uncompressed) + " prefix=" + std::to_string(prefix);
          return -1;
        }
        page += vals;
      } else if (meta.codec == 0) {
        page.append(reinterpret_cast<const char *>(valbody), valbytes);
      } else {
        error = "unsupported parquet compression codec";
        return -1;
      }
    } else if (meta.codec == 1) {
      if (snappy_uncompress(body, static_cast<size_t>(compressed), page,
                            static_cast<size_t>(uncompressed)) != 0) {
        error = "snappy decompress failed (other) page=" + std::to_string(page_type)
                + " c=" + std::to_string(compressed) + " u=" + std::to_string(uncompressed)
                + " off=" + std::to_string(offset);
        return -1;
      }
    } else if (meta.codec == 0) {
      page.assign(reinterpret_cast<const char *>(body), static_cast<size_t>(compressed));
    } else {
      error = "unsupported parquet compression codec";
      return -1;
    }
    offset += static_cast<int64_t>(header_len) + compressed;

    const unsigned char *pdata = reinterpret_cast<const unsigned char *>(page.data());
    const size_t psize = page.size();
    if (page_type == 2) {  // DICTIONARY_PAGE
      dictionary.clear();
      if (plain_decode_values(pdata, psize, meta.type,
                              static_cast<size_t>(dict_num_values), dictionary) != 0) {
        error = "bad dictionary page";
        return -1;
      }
    } else if (page_type == 0) {  // DATA_PAGE v1
      size_t pos = 0;
      std::vector<uint32_t> defs;
      if (optional) {
        if (def_encoding == 3 /* RLE */) {
          if (pos + 4 > psize) {
            error = "bad definition levels";
            return -1;
          }
          const uint32_t len = read_le32(pdata + pos);
          pos += 4;
          if (rle_hybrid_decode(pdata + pos, len, 1, static_cast<size_t>(num_values), defs) != 0) {
            error = "bad definition levels";
            return -1;
          }
          pos += len;
        } else {
          error = "unsupported definition level encoding";
          return -1;
        }
      }
      size_t present = 0;
      for (int32_t i = 0; i < num_values; ++i) {
        const bool ok = optional ? (i < static_cast<int32_t>(defs.size()) && defs[static_cast<size_t>(i)] == 1)
                                 : true;
        out.present.push_back(ok ? 1 : 0);
        if (ok) { ++present; }
      }
      std::vector<std::string> values;
      if (encoding == 0) {  // PLAIN
        if (plain_decode_values(pdata + pos, psize - pos, meta.type, present, values) != 0) {
          error = "bad plain encoded values";
          return -1;
        }
      } else if (encoding == 8 || encoding == 2) {  // RLE_DICTIONARY / PLAIN_DICTIONARY
        if (pos >= psize) {
          error = "bad dictionary indices";
          return -1;
        }
        const int width = pdata[pos++];
        std::vector<uint32_t> idx;
        if (rle_hybrid_decode(pdata + pos, psize - pos, width, present, idx) != 0) {
          error = "bad dictionary indices";
          return -1;
        }
        for (size_t i = 0; i < idx.size(); ++i) {
          if (idx[i] >= dictionary.size()) {
            error = "dictionary index out of range";
            return -1;
          }
          values.push_back(dictionary[idx[i]]);
        }
      } else {
        error = "unsupported parquet encoding";
        return -1;
      }
      out.values.insert(out.values.end(), values.begin(), values.end());
    } else if (page_type == 3) {  // DATA_PAGE v2
      const size_t rep_len = static_cast<size_t>(v2_rep_len);
      const size_t def_len = static_cast<size_t>(v2_def_len);
      if (rep_len + def_len > psize) {
        error = "bad v2 levels";
        return -1;
      }
      std::vector<uint32_t> defs;
      if (optional && def_len > 0) {
        if (rle_hybrid_decode(pdata + rep_len, def_len, 1,
                              static_cast<size_t>(v2_num_values), defs) != 0) {
          error = "bad v2 definition levels";
          return -1;
        }
      }
      size_t present = 0;
      for (int32_t i = 0; i < v2_num_values; ++i) {
        const bool ok = optional ? (i < static_cast<int32_t>(defs.size()) && defs[static_cast<size_t>(i)] == 1)
                                 : true;
        out.present.push_back(ok ? 1 : 0);
        if (ok) { ++present; }
      }
      const unsigned char *vdata = pdata + rep_len + def_len;
      const size_t vsize = psize - rep_len - def_len;
      std::vector<std::string> values;
      if (encoding == 0) {
        if (plain_decode_values(vdata, vsize, meta.type, present, values) != 0) {
          error = "bad v2 plain values";
          return -1;
        }
      } else if (encoding == 8 || encoding == 2) {
        if (vsize < 1) {
          error = "bad v2 dictionary indices";
          return -1;
        }
        const int width = vdata[0];
        std::vector<uint32_t> idx;
        if (rle_hybrid_decode(vdata + 1, vsize - 1, width, present, idx) != 0) {
          error = "bad v2 dictionary indices";
          return -1;
        }
        for (size_t i = 0; i < idx.size(); ++i) {
          if (idx[i] >= dictionary.size()) {
            error = "dictionary index out of range";
            return -1;
          }
          values.push_back(dictionary[idx[i]]);
        }
      } else {
        error = "unsupported v2 encoding";
        return -1;
      }
      out.values.insert(out.values.end(), values.begin(), values.end());
    }
  }
  if (out.present.size() < static_cast<size_t>(meta.num_values)) {
    out.present.resize(static_cast<size_t>(meta.num_values), 0);
  }
  return 0;
}

inline ColumnType map_physical_type(int32_t type)
{
  switch (type) {
    case 0: return ColumnType::BOOLEAN;
    case 1:
    case 2: return ColumnType::BIGINT;
    case 4:
    case 5: return ColumnType::DOUBLE;
    default: return ColumnType::VARCHAR;
  }
}

}  // namespace pq

inline int read_parquet_as_json(const std::string &path, Schema &schema,
                                std::string &json, std::string &error)
{
  std::string file;
  {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
      error = "cannot open file: " + path;
      return -1;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    file = ss.str();
  }
  if (file.size() < 12 || file.compare(0, 4, "PAR1") != 0
      || file.compare(file.size() - 4, 4, "PAR1") != 0) {
    error = "not a parquet file";
    return -1;
  }
  const unsigned char *base = reinterpret_cast<const unsigned char *>(file.data());
  const uint32_t footer_len = pq::read_le32(base + file.size() - 8);
  if (footer_len + 12 > file.size()) {
    error = "bad parquet footer";
    return -1;
  }
  pq::FileMeta meta;
  if (pq::parse_file_meta(base + file.size() - 8 - footer_len, footer_len, meta) != 0) {
    error = "failed to parse parquet metadata";
    return -1;
  }
  if (meta.schema.empty()) {
    error = "empty parquet schema";
    return -1;
  }
  // The first element is the root group; its children are the leaf columns.
  std::vector<pq::SchemaElem> leaves;
  for (size_t i = 1; i < meta.schema.size(); ++i) {
    if (meta.schema[i].num_children == 0) {
      leaves.push_back(meta.schema[i]);
    }
  }
  if (leaves.empty()) {
    error = "parquet schema has no leaf columns";
    return -1;
  }
  schema.clear();
  for (size_t c = 0; c < leaves.size(); ++c) {
    Column col;
    col.name = leaves[c].name;
    col.type = pq::map_physical_type(leaves[c].type);
    col.nullable = leaves[c].repetition == 1;
    col.length = 1024;
    schema.push_back(col);
  }
  std::vector<pq::ColumnData> data(leaves.size());
  for (size_t rg = 0; rg < meta.row_groups.size(); ++rg) {
    const std::vector<pq::ColMeta> &cols = meta.row_groups[rg];
    for (size_t c = 0; c < cols.size() && c < leaves.size(); ++c) {
      if (pq::decode_column(file, cols[c], schema[c].nullable, data[c], error) != 0) {
        return -1;
      }
    }
  }
  size_t rows = static_cast<size_t>(meta.num_rows);
  if (rows == 0) {
    for (size_t c = 0; c < data.size(); ++c) {
      rows = std::max(rows, data[c].present.size());
    }
  }
  const size_t total_rows = data.empty() ? rows : data[0].present.size();
  size_t value_pos = 0;
  json.clear();
  json.push_back('[');
  for (size_t r = 0; r < total_rows; ++r) {
    if (r > 0) { json.push_back(','); }
    json.push_back('{');
    for (size_t c = 0; c < schema.size(); ++c) {
      if (c > 0) { json.push_back(','); }
      json.push_back('"');
      json += schema[c].name;
      json += "\":";
      const bool present = r < data[c].present.size() && data[c].present[r] != 0;
      if (!present) {
        json += "null";
      } else {
        const size_t index = std::count(data[c].present.begin(), data[c].present.begin() + r, 1);
        const std::string &text = index < data[c].values.size() ? data[c].values[index] : std::string();
        if (schema[c].type == ColumnType::BIGINT || schema[c].type == ColumnType::DOUBLE) {
          json += text.empty() ? "0" : text;
        } else {
          json.push_back('"');
          json += text;
          json.push_back('"');
        }
      }
    }
    json.push_back('}');
    ++value_pos;
  }
  json.push_back(']');
  return 0;
}

// ---------------------------------------------------------------------------
// Helpers shared by the resolver and DESCRIBE SELECT handling.
// ---------------------------------------------------------------------------
inline int read_file_content(const std::string &path, std::string &content, std::string &error)
{
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    error = "cannot open file: " + path;
    return -1;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  content = ss.str();
  return 0;
}

// Mirrors ObDMLResolver::fsql_flatten_args: a T_LINK_NODE chain stores the
// previous link in child 0 and the current argument in child 1.
inline void collect_args(const ParseNode *node, std::vector<const ParseNode *> &args)
{
  args.clear();
  while (OB_NOT_NULL(node)) {
    if (T_LINK_NODE == node->type_ && 2 == node->num_child_
        && OB_NOT_NULL(node->children_[1])) {
      args.push_back(node->children_[1]);
      node = node->children_[0];
    } else {
      args.push_back(node);
      node = NULL;
    }
  }
  std::reverse(args.begin(), args.end());
}

// Reads the file referenced by a T_FILE_SQL_EXPRESSION node and fills schema.
inline int file_sql_schema_from_node(const ParseNode &file_node, Schema &schema, std::string &error)
{
  int ret = 0;
  const ParseNode *type_node = NULL;
  const ParseNode *args_node = NULL;
  std::vector<const ParseNode *> args;
  if (T_FILE_SQL_EXPRESSION != file_node.type_ || 3 != file_node.num_child_
      || OB_ISNULL(type_node = file_node.children_[0])
      || OB_ISNULL(args_node = file_node.children_[1])) {
    error = "invalid file table function";
    ret = -1;
  } else {
    collect_args(args_node, args);
    if (args.empty() || T_VARCHAR != args[0]->type_) {
      error = "file path must be a string literal";
      ret = -1;
    } else {
      const int64_t file_type = type_node->value_;
      ReadOptions options;
      options.type = (0 == file_type) ? FileType::CSV : FileType::JSONL;
      ObString format;
      for (size_t i = 1; 0 == ret && i < args.size(); ++i) {
        const ParseNode *arg = args[i];
        if (T_VARCHAR == arg->type_) {
          format.assign_ptr(arg->str_value_, static_cast<int32_t>(arg->str_len_));
        } else if (T_OP_EQ == arg->type_ && 2 == arg->num_child_
                   && OB_NOT_NULL(arg->children_[0]) && OB_NOT_NULL(arg->children_[1])) {
          const ParseNode *name_node = arg->children_[0];
          const ParseNode *value_node = arg->children_[1];
          const ObString name(static_cast<int32_t>(name_node->str_len_), name_node->str_value_);
          if (0 == name.case_compare("header")) {
            if (T_BOOL == value_node->type_ || T_INT == value_node->type_) {
              options.has_header = (0 != value_node->value_);
            }
          } else if (0 == name.case_compare("sample_size")) {
            if (-1 != value_node->value_) {
              error = "only sample_size = -1 is supported";
              ret = -1;
            }
          } else if (0 == name.case_compare("format") && T_VARCHAR == value_node->type_) {
            format.assign_ptr(value_node->str_value_, static_cast<int32_t>(value_node->str_len_));
          }
        }
      }
      if (0 == ret && !format.empty()) {
        if (0 == file_type && 0 != format.case_compare("csv")) {
          error = "unsupported csv format";
          ret = -1;
        } else if (1 == file_type && 0 != format.case_compare("newline_delimited")
                   && 0 != format.case_compare("ndjson")) {
          error = "unsupported json format";
          ret = -1;
        }
      }
      const std::string path(args[0]->str_value_, static_cast<size_t>(args[0]->str_len_));
      std::string json;
      if (0 == ret) {
        if (3 == file_type) {
          if (0 != glob_files_as_json(path, json, error)) { ret = -1; }
        } else if (2 == file_type) {
          if (0 != read_parquet_as_json(path, schema, json, error)) { ret = -1; }
        } else if (0 != read_file_as_json(path, options, schema, json, error)) {
          ret = -1;
        }
      }
      if (0 == ret && 3 == file_type) {
        schema.clear();
        Column file_col;
        file_col.name = "file";
        file_col.type = ColumnType::VARCHAR;
        file_col.nullable = false;
        file_col.length = 1024;
        schema.push_back(file_col);
      }
    }
  }
  return ret;
}

inline const ParseNode *find_file_sql_node(const ParseNode &node)
{
  const ParseNode *found = NULL;
  if (T_FILE_SQL_EXPRESSION == node.type_) {
    found = &node;
  } else {
    for (int64_t i = 0; NULL == found && i < node.num_child_; ++i) {
      if (OB_NOT_NULL(node.children_[i])) {
        found = find_file_sql_node(*node.children_[i]);
      }
    }
  }
  return found;
}

}  // namespace filesql
}  // namespace sql
}  // namespace oceanbase
#endif  // OCEANBASE_SQL_RESOLVER_DML_OB_FILE_SQL_READER_H_
