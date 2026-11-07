/**
 * Copyright (c) 2021 OceanBase
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OCEANBASE_OBSERVER_AI_COLUMN_WORKER_H_
#define OCEANBASE_OBSERVER_AI_COLUMN_WORKER_H_

#include <atomic>
#include <string>
#include <thread>
#include <unistd.h>

#include <curl/curl.h>

#include "lib/oblog/ob_log.h"
#include "lib/string/ob_sql_string.h"
#include "share/ob_server_struct.h"
#include "share/ob_define.h"
#include "share/ai_service/ob_ai_column_state.h"

namespace oceanbase
{
namespace observer
{

// Header-only background worker that fulfils AI column tasks. It polls
// oceanbase.__all_ai_column_task and, for every pending row, calls the model
// endpoint and writes the answer back to the user table.
class ObAIColumnWorker
{
public:
  ObAIColumnWorker() : stop_(false), started_(false) {}
  ~ObAIColumnWorker() { wait(); }

  int start()
  {
    int ret = common::OB_SUCCESS;
    if (started_) {
    } else {
      stop_ = false;
      try {
        thread_ = std::thread(&ObAIColumnWorker::run, this);
        started_ = true;
        LOG_INFO("ai column worker started");
      } catch (...) {
        ret = common::OB_ERR_UNEXPECTED;
        LOG_INFO("fail to start ai column worker thread", K(ret));
      }
    }
    return ret;
  }

  void wait()
  {
    if (started_ && thread_.joinable()) {
      stop_ = true;
      thread_.join();
      started_ = false;
    }
  }

private:
  struct CurlBuf
  {
    std::string data;
  };

  static size_t curl_write_cb(void *ptr, size_t size, size_t nmemb, void *userdata)
  {
    CurlBuf *buf = static_cast<CurlBuf *>(userdata);
    buf->data.append(static_cast<char *>(ptr), size * nmemb);
    return size * nmemb;
  }

  static std::string json_escape(const std::string &in)
  {
    std::string out;
    for (char c : in) {
      if ('"' == c) { out += "\\\""; }
      else if ('\\' == c) { out += "\\\\"; }
      else if ('\n' == c) { out += "\\n"; }
      else if ('\r' == c) { out += "\\r"; }
      else if ('\t' == c) { out += "\\t"; }
      else { out += c; }
    }
    return out;
  }

  static std::string sql_escape(const std::string &in)
  {
    std::string out;
    for (char c : in) {
      if ('\'' == c || '\\' == c) { out += '\\'; }
      out += c;
    }
    return out;
  }

  // Extract the first embedding array from an OpenAI compatible response body.
  static std::string pick_embedding(const std::string &body)
  {
    std::string out;
    size_t pos = body.find("\"embedding\"");
    if (std::string::npos == pos) {
      return out;
    }
    size_t open = body.find('[', pos);
    if (std::string::npos == open) {
      return out;
    }
    size_t close = body.find(']', open);
    if (std::string::npos == close) {
      return out;
    }
    out.reserve(close - open + 1);
    out += '[';
    for (size_t i = open + 1; i < close; ++i) {
      const char c = body[i];
      if (' ' == c || '\n' == c || '\t' == c) {
        continue;
      }
      out += c;
    }
    out += ']';
    return out;
  }

  // Extract "content" from an OpenAI compatible response body.
  static std::string pick_content(const std::string &body)
  {
    std::string out;
    size_t pos = body.find("\"content\"");
    if (std::string::npos == pos) {
      pos = 0;
      size_t brace = body.find('{');
      if (std::string::npos != brace) { out = body.substr(brace, std::string::npos); }
      return out;
    }
    size_t colon = body.find(':', pos);
    size_t i = colon;
    while (std::string::npos != i && i < body.size() && body[i] != '"') { ++i; }
    if (std::string::npos == i || i >= body.size()) { return out; }
    ++i;
    while (i < body.size() && body[i] != '"') {
      if ('\\' == body[i] && i + 1 < body.size()) {
        char n = body[i + 1];
        if ('n' == n) { out += '\n'; }
        else if ('t' == n) { out += '\t'; }
        else if ('r' == n) { out += '\r'; }
        else { out += n; }
        i += 2;
      } else {
        out += body[i++];
      }
    }
    return out;
  }

  void run()
  {
    int64_t times = 0;
    while (!stop_) {
      usleep(100 * 1000);
      int ret = do_round();
      ++times;
      if (common::OB_SUCCESS != ret) {
        LOG_INFO("ai column worker round failed", K(ret));
      }
      if (0 == times % 100) {
        LOG_INFO("ai column worker heartbeat", K(times));
      }
    }
  }

  int do_round()
  {
    int ret = common::OB_SUCCESS;
    if (NULL == GCTX.sql_proxy_) {
      ret = common::OB_NOT_INIT;
    } else {
      std::vector<PreparedTask> batch;
      SMART_VAR(ObMySQLProxy::MySQLResult, res) {
        sqlclient::ObMySQLResult *result = NULL;
        const char *sql =
            "select cast(table_id as char) as tid, rowkey_predicate, column_name, "
            "cast(request_id as char) as rid, cast(retry_count as char) as rc "
            "from oceanbase.__all_ai_column_task "
            "where (state = 0 or state = 3) and eligible_ts <= now(6) "
            "order by eligible_ts limit 4";
        if (OB_FAIL(GCTX.sql_proxy_->read(res, sql))) {
        } else if (NULL == (result = res.get_result())) {
          ret = common::OB_ERR_UNEXPECTED;
        } else {
          while (OB_SUCC(ret)) {
            ret = result->next();
            if (common::OB_ITER_END == ret) { ret = common::OB_SUCCESS; break; }
            if (OB_FAIL(ret)) { break; }
            common::ObString tid;
            common::ObString rowkey;
            common::ObString col;
            common::ObString rid;
            common::ObString rc;
            if (OB_FAIL(result->get_varchar("tid", tid))
                || OB_FAIL(result->get_varchar("rowkey_predicate", rowkey))
                || OB_FAIL(result->get_varchar("column_name", col))
                || OB_FAIL(result->get_varchar("rid", rid))
                || OB_FAIL(result->get_varchar("rc", rc))) {
              ret = common::OB_SUCCESS;
              continue;
            }
            LOG_INFO("ai column worker picked task", K(tid), K(col));
            std::string s_tid(tid.ptr(), tid.length());
            std::string s_rowkey(rowkey.ptr(), rowkey.length());
            std::string s_col(col.ptr(), col.length());
            std::string s_rc(rc.ptr(), rc.length());
            std::string s_rid(rid.ptr(), rid.length());
            PreparedTask prepared;
            int tmp = handle_task(s_tid, s_rowkey, s_col, s_rc, s_rid, prepared);
            if (common::OB_SUCCESS != tmp) {
              LOG_INFO("ai column worker handle task failed", K(tmp));
            } else if (prepared.ready) {
              batch.push_back(prepared);
            }
          }
        }
      }
      apply_prepared(batch);
    }
    return ret;
  }

  // Number of tasks that are still queued for the column, -1 when unknown.
  int64_t count_tasks(const std::string &table_id, const std::string &column_name)
  {
    int ret = common::OB_SUCCESS;
    int64_t remaining = -1;
    common::ObSqlString sql;
    SMART_VAR(ObMySQLProxy::MySQLResult, res) {
      sqlclient::ObMySQLResult *result = NULL;
      if (NULL == GCTX.sql_proxy_) {
      } else if (OB_FAIL(sql.assign_fmt("select count(*) as c from oceanbase.__all_ai_column_task "
                                        "where table_id = %s and column_name = '%s'",
                                        table_id.c_str(), sql_escape(column_name).c_str()))) {
      } else if (OB_FAIL(GCTX.sql_proxy_->read(res, sql.ptr()))) {
      } else if (NULL == (result = res.get_result())) {
      } else if (OB_SUCC(result->next())) {
        int64_t count = 0;
        if (OB_SUCC(result->get_int("c", count))) {
          remaining = count;
        }
      }
    }
    return remaining;
  }

  int64_t fetch_max_retries(const std::string &table_id, const std::string &column_name)
  {
    int ret = common::OB_SUCCESS;
    int64_t max_retries = 0;
    std::string state_col = "__ai_state_" + column_name;
    common::ObSqlString sql;
    SMART_VAR(ObMySQLProxy::MySQLResult, res) {
      sqlclient::ObMySQLResult *result = NULL;
      if (NULL == GCTX.sql_proxy_) {
      } else if (OB_FAIL(sql.assign_fmt("select comment from oceanbase.__all_column "
                                        "where table_id = %s and column_name = '%s'",
                                        table_id.c_str(), sql_escape(state_col).c_str()))) {
      } else if (OB_FAIL(GCTX.sql_proxy_->read(res, sql.ptr()))) {
      } else if (NULL == (result = res.get_result())) {
      } else if (OB_SUCC(result->next())) {
        common::ObString cmt;
        if (OB_SUCC(result->get_varchar("comment", cmt))) {
          std::string text(cmt.ptr(), cmt.length());
          size_t pos = text.find("ai:max_retries=");
          if (std::string::npos != pos) {
            max_retries = atoll(text.c_str() + pos + 15);
          }
        }
      }
    }
    return max_retries;
  }

  void drop_task(const std::string &table_id,
                 const std::string &rowkey,
                 const std::string &column_name)
  {
    common::ObSqlString del;
    if (common::OB_SUCCESS == del.assign_fmt(
            "delete from oceanbase.__all_ai_column_task where table_id = %s "
            "and rowkey_predicate = '%s' and column_name = '%s'",
            table_id.c_str(), sql_escape(rowkey).c_str(), column_name.c_str())) {
      int64_t delcnt = 0;
      int tmp = GCTX.sql_proxy_->write(del.ptr(), delcnt);
      LOG_INFO("ai column worker dropped stale task", K(tmp), K(delcnt));
    }
  }

  void defer_task(const std::string &table_id,
                  const std::string &rowkey,
                  const std::string &column_name)
  {
    common::ObSqlString upd;
    if (common::OB_SUCCESS == upd.assign_fmt(
            "update oceanbase.__all_ai_column_task "
            "set eligible_ts = date_add(now(6), interval 1 second) "
            "where table_id = %s and rowkey_predicate = '%s' and column_name = '%s'",
            table_id.c_str(), sql_escape(rowkey).c_str(), column_name.c_str())) {
      int64_t affected = 0;
      int tmp = GCTX.sql_proxy_->write(upd.ptr(), affected);
      LOG_INFO("ai column worker defer task", K(tmp), K(affected));
    }
  }

  struct PreparedTask
  {
    bool ready;
    std::string table_id;
    std::string rowkey;
    std::string column_name;
    std::string db_name;
    std::string table_name;
    std::string answer;
    std::string request_id;
    std::string base_value;
    PreparedTask() : ready(false) {}
  };

  int handle_task(const std::string &table_id,
                  const std::string &rowkey,
                  const std::string &column_name,
                  const std::string &retry_count,
                  const std::string &request_id,
                  PreparedTask &prepared)
  {
    int ret = common::OB_SUCCESS;
    std::string db_name;
    std::string table_name;
    std::string expr;
    std::string url;
    std::string access_key;

    // user table name and database name
    {
      common::ObSqlString sql;
      SMART_VAR(ObMySQLProxy::MySQLResult, res) {
        sqlclient::ObMySQLResult *result = NULL;
        if (OB_FAIL(sql.assign_fmt("select t.table_name, d.database_name from oceanbase.__all_table t "
                                   "join oceanbase.__all_database d on d.database_id = t.database_id "
                                   "where t.table_id = %s", table_id.c_str()))) {
        } else if (OB_FAIL(GCTX.sql_proxy_->read(res, sql.ptr()))) {
        } else if (NULL == (result = res.get_result())) {
          ret = common::OB_ERR_UNEXPECTED;
        } else if (OB_FAIL(result->next())) {
          ret = common::OB_SUCCESS;
        } else {
          common::ObString t;
          common::ObString d;
          if (OB_SUCC(result->get_varchar("table_name", t))
              && OB_SUCC(result->get_varchar("database_name", d))) {
            table_name.assign(t.ptr(), t.length());
            db_name.assign(d.ptr(), d.length());
          }
        }
      }
    }

    LOG_INFO("ai column worker resolved table", K(db_name.c_str()), K(table_name.c_str()),
             K(column_name.c_str()));
    if (table_name.empty()) {
      share::ai_column_state().mark(atoll(table_id.c_str()), column_name.c_str(),
                                            static_cast<int64_t>(column_name.size()),
                                          share::ObAIColumnState::FAILED);
      drop_task(table_id, rowkey, column_name);
      return common::OB_SUCCESS;
    }

    // request column comment carries "ai:expr=AI_COMPLETE('m', concat('x', content))"
    bool integer_hint = false;
    bool json_hint = false;
    if (!table_name.empty()) {
      common::ObSqlString sql;
      std::string req = "__ai_req_" + column_name;
      SMART_VAR(ObMySQLProxy::MySQLResult, res) {
        sqlclient::ObMySQLResult *result = NULL;
        if (OB_FAIL(sql.assign_fmt("select comment from oceanbase.__all_column where table_id = %s "
                                   "and column_name = '%s'", table_id.c_str(), req.c_str()))) {
        } else if (OB_FAIL(GCTX.sql_proxy_->read(res, sql.ptr()))) {
        } else if (NULL == (result = res.get_result())) {
          ret = common::OB_ERR_UNEXPECTED;
        } else if (OB_SUCC(result->next())) {
          common::ObString cmt;
          if (OB_SUCC(result->get_varchar("comment", cmt))) {
            std::string text(cmt.ptr(), cmt.length());
            integer_hint = (std::string::npos != text.find("base-10 integer"));
            json_hint = (std::string::npos != text.find("valid JSON value"));
            size_t pos = text.find("ai:expr=");
            if (std::string::npos != pos) {
              std::string body = text.substr(pos + 8);
              size_t pct = body.find('%');
              if (std::string::npos != pct) { body = body.substr(0, pct); }
              expr = body;
            }
          }
        }
      }
      ret = common::OB_SUCCESS;
    }

    // parse model name and the "concat('<prefix>', content)" input
    std::string model;
    std::string prefix;
    {
      size_t open = expr.find('(');
      size_t comma = (std::string::npos == open) ? std::string::npos : expr.find(',', open + 1);
      if (std::string::npos != comma) {
        std::string m = expr.substr(open + 1, comma - open - 1);
        size_t b = m.find_first_not_of(" '");
        size_t e = m.find_last_not_of(" '");
        if (std::string::npos != b && std::string::npos != e && e >= b) {
          model = m.substr(b, e - b + 1);
        }
        std::string input = expr.substr(comma + 1);
        size_t q1 = input.find('\'');
        if (std::string::npos != q1) {
          size_t q2 = input.find('\'', q1 + 1);
          if (std::string::npos != q2) { prefix = input.substr(q1 + 1, q2 - q1 - 1); }
        }
      }
    }

    // endpoint
    if (!model.empty()) {
      common::ObSqlString sql;
      SMART_VAR(ObMySQLProxy::MySQLResult, res) {
        sqlclient::ObMySQLResult *result = NULL;
        if (OB_FAIL(sql.assign_fmt("select url, access_key from oceanbase.__all_ai_model_endpoint "
                                   "where ai_model_name = '%s' limit 1", sql_escape(model).c_str()))) {
        } else if (OB_FAIL(GCTX.sql_proxy_->read(res, sql.ptr()))) {
        } else if (NULL == (result = res.get_result())) {
          ret = common::OB_ERR_UNEXPECTED;
        } else if (OB_SUCC(result->next())) {
          common::ObString u;
          common::ObString k;
          if (OB_SUCC(result->get_varchar("url", u))) {
            url.assign(u.ptr(), u.length());
          }
          if (OB_SUCC(result->get_varchar("access_key", k))) {
            access_key.assign(k.ptr(), k.length());
          }
        }
      }
      ret = common::OB_SUCCESS;
    }

    // input text from the user row
    std::string user_text;
    std::string prepared_base_value;
    if (!url.empty()) {
      common::ObSqlString sql;
      SMART_VAR(ObMySQLProxy::MySQLResult, res) {
        sqlclient::ObMySQLResult *result = NULL;
        if (OB_FAIL(sql.assign_fmt("select content from `%s`.`%s` where %s limit 1",
                                   db_name.c_str(), table_name.c_str(), rowkey.c_str()))) {
        } else if (OB_FAIL(GCTX.sql_proxy_->read(res, sql.ptr()))) {
        } else if (NULL == (result = res.get_result())) {
          ret = common::OB_ERR_UNEXPECTED;
        } else if (OB_SUCC(result->next())) {
          common::ObString v;
          if (OB_SUCC(result->get_varchar("content", v))) {
            user_text = prefix + std::string(v.ptr(), v.length());
            prepared_base_value = std::string(v.ptr(), v.length());
          }
        }
      }
      ret = common::OB_SUCCESS;
    }

    LOG_INFO("ai column worker prepared", K(expr.c_str()), K(model.c_str()), K(url.c_str()),
             K(user_text.c_str()));
    if (url.empty()) {
      share::ai_column_state().mark(atoll(table_id.c_str()), column_name.c_str(),
                                   static_cast<int64_t>(column_name.size()),
                                   share::ObAIColumnState::FAILED);
      drop_task(table_id, rowkey, column_name);
      return common::OB_SUCCESS;
    }
    if (user_text.empty()) {
      defer_task(table_id, rowkey, column_name);
      return common::OB_SUCCESS;
    }
    std::string answer;
    if (!url.empty() && !user_text.empty()) {
      CURL *curl = curl_easy_init();
      if (NULL != curl) {
        const char *sys_prompt =
            integer_hint
                ? "Return exactly one base-10 integer value when the column type is an integer."
                : (json_hint ? "Return exactly one valid JSON value when the column type is JSON."
                             : "Answer with one short plain-text line.");
        const bool is_embed = (std::string::npos != expr.find("AI_EMBED"));
        std::string body;
        if (is_embed) {
          body = "{\"model\":\"" + json_escape(model) + "\",\"input\":\"" + json_escape(user_text)
                 + "\"}";
        } else {
          body = "{\"model\":\"" + json_escape(model) + "\",\"messages\":[{\"role\":\"system\",\"content\":\"" + std::string(sys_prompt) + "\"},{\"role\":\"user\",\"content\":\"" + json_escape(user_text) + "\"}]}";
        }
        std::string auth = "Authorization: Bearer " + access_key;
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        headers = curl_slist_append(headers, auth.c_str());
        CurlBuf buf;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &ObAIColumnWorker::curl_write_cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
        CURLcode cc = curl_easy_perform(curl);
        long code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        if (CURLE_OK != cc || code >= 400) {
          LOG_INFO("ai column worker request failed", K(cc), K(code));
          const int64_t retried = atoll(retry_count.c_str()) + 1;
          const int64_t max_retries = fetch_max_retries(table_id, column_name);
          if (retried > max_retries) {
            share::ai_column_state().mark(atoll(table_id.c_str()), column_name.c_str(),
                                                  static_cast<int64_t>(column_name.size()),
                                          share::ObAIColumnState::FAILED);
            drop_task(table_id, rowkey, column_name);
          } else {
            common::ObSqlString upd;
            if (OB_SUCC(upd.assign_fmt("update oceanbase.__all_ai_column_task set state = 3, "
                                       "retry_count = retry_count + 1, "
                                       "eligible_ts = date_add(now(6), interval 1 second) "
                                       "where table_id = %s and rowkey_predicate = '%s' "
                                       "and column_name = '%s'",
                                       table_id.c_str(), sql_escape(rowkey).c_str(),
                                       column_name.c_str()))) {
              int64_t affected = 0;
              int tmp = GCTX.sql_proxy_->write(upd.ptr(), affected);
              LOG_INFO("ai column worker mark retry", K(tmp), K(affected), K(retried),
                       K(max_retries));
            }
          }
          return common::OB_SUCCESS;
        }
        answer = is_embed ? pick_embedding(buf.data) : pick_content(buf.data);
      }
    }

    if (!answer.empty()) {
      prepared.table_id = table_id;
      prepared.rowkey = rowkey;
      prepared.column_name = column_name;
      prepared.db_name = db_name;
      prepared.table_name = table_name;
      prepared.answer = answer;
      prepared.request_id = request_id;
      prepared.base_value = prepared_base_value;
      prepared.ready = true;
    }
    return common::OB_SUCCESS;
  }

  // All AI columns of the same row are applied with a single UPDATE so readers
  // never observe a row where only part of the AI columns is filled.
  void apply_prepared(std::vector<PreparedTask> &batch)
  {
    int ret = common::OB_SUCCESS;
    for (size_t i = 0; i < batch.size(); ++i) {
      if (!batch[i].ready) {
        continue;
      }
      std::string columns_sql;
      std::string delete_sql;
      std::string db_name = batch[i].db_name;
      std::string table_name = batch[i].table_name;
      std::string rowkey = batch[i].rowkey;
      std::string table_id = batch[i].table_id;
      for (size_t j = i; j < batch.size(); ++j) {
        if (!batch[j].ready || batch[j].db_name != db_name || batch[j].table_name != table_name
            || batch[j].rowkey != rowkey || batch[j].table_id != table_id) {
          continue;
        }
        if (!columns_sql.empty()) {
          columns_sql += ", ";
        }
        columns_sql += "`" + batch[j].column_name + "` = '" + sql_escape(batch[j].answer) + "'";
        if (!delete_sql.empty()) {
          delete_sql += " or ";
        }
        delete_sql += "column_name = '" + batch[j].column_name + "'";
        batch[j].ready = false;
      }
      // A newer task for the same row/column means this answer is stale.
      bool superseded = false;
      bool saw_current = false;
      {
        common::ObSqlString cur;
        SMART_VAR(ObMySQLProxy::MySQLResult, rres) {
          sqlclient::ObMySQLResult *rresult = NULL;
          if (OB_SUCC(cur.assign_fmt("select content from `%s`.`%s` where %s limit 1",
                                     batch[i].db_name.c_str(), batch[i].table_name.c_str(),
                                     rowkey.c_str()))) {
            if (OB_SUCC(GCTX.sql_proxy_->read(rres, cur.ptr()))) {
              rresult = rres.get_result();
              if (NULL != rresult && OB_SUCC(rresult->next())) {
                common::ObString cv;
                if (OB_SUCC(rresult->get_varchar("content", cv))) {
                  saw_current = true;
                  superseded = (batch[i].base_value != std::string(cv.ptr(), cv.length()));
                  LOG_INFO("ai column guard check", K(batch[i].base_value.c_str()),
                           K(std::string(cv.ptr(), cv.length()).c_str()), K(superseded));
                }
              }
            }
          }
        }
      }
      for (size_t k = i; !superseded && k < batch.size(); ++k) {
        (void)k;
        break;
      }
      {
        common::ObSqlString chk;
        SMART_VAR(ObMySQLProxy::MySQLResult, cres) {
          sqlclient::ObMySQLResult *cresult = NULL;
          if (OB_SUCC(chk.assign_fmt(
                  "select count(*) as newer from oceanbase.__all_ai_column_task "
                  "where table_id = %s and rowkey_predicate = '%s' and request_id > %s",
                  table_id.c_str(), sql_escape(rowkey).c_str(),
                  batch[i].request_id.c_str()))) {
            if (OB_SUCC(GCTX.sql_proxy_->read(cres, chk.ptr()))) {
              cresult = cres.get_result();
              if (NULL != cresult && OB_SUCC(cresult->next())) {
                int64_t newer = 0;
                if (OB_SUCC(cresult->get_int("newer", newer)) && newer > 0) {
                  superseded = true;
                }
              }
            }
          }
        }
      }
      if (!saw_current) {
        LOG_INFO("ai column guard saw no row", K(rowkey.c_str()));
      }
      if (superseded) {
        LOG_INFO("ai column worker skip stale answer", K(rowkey.c_str()),
                 K(batch[i].request_id.c_str()));
        std::string stale_delete;
        for (size_t k = i; k < batch.size(); ++k) {
          if (!batch[k].ready || batch[k].request_id != batch[i].request_id
              || batch[k].rowkey != rowkey || batch[k].table_id != table_id) {
            continue;
          }
          if (!stale_delete.empty()) {
            stale_delete += " or ";
          }
          stale_delete += "column_name = '" + batch[k].column_name + "'";
          batch[k].ready = false;
        }
        common::ObSqlString del;
        if (OB_SUCC(del.assign_fmt("delete from oceanbase.__all_ai_column_task where table_id = %s "
                                   "and rowkey_predicate = '%s' and request_id = %s and (%s)",
                                   table_id.c_str(), sql_escape(rowkey).c_str(),
                                   batch[i].request_id.c_str(), stale_delete.c_str()))) {
          int64_t delcnt = 0;
          int tmp2 = GCTX.sql_proxy_->write(del.ptr(), delcnt);
          LOG_INFO("ai column worker stale task dropped", K(tmp2), K(delcnt));
        }
        continue;
      }
      common::ObSqlString upd;
      if (OB_SUCC(upd.assign_fmt("update `%s`.`%s` set %s where %s", db_name.c_str(),
                                 table_name.c_str(), columns_sql.c_str(), rowkey.c_str()))) {
        int64_t affected = 0;
        int tmp = GCTX.sql_proxy_->write(upd.ptr(), affected);
        LOG_INFO("ai column worker write back", K(tmp), K(affected), K(columns_sql.c_str()));
        common::ObSqlString del;
        if (OB_SUCC(del.assign_fmt("delete from oceanbase.__all_ai_column_task where table_id = %s "
                                   "and rowkey_predicate = '%s' and request_id = %s and (%s)",
                                   table_id.c_str(), sql_escape(rowkey).c_str(),
                                   batch[i].request_id.c_str(), delete_sql.c_str()))) {
          int64_t delcnt = 0;
          int tmp2 = GCTX.sql_proxy_->write(del.ptr(), delcnt);
          if (0 == tmp2 && delcnt > 0) {
            // one batch may carry several AI columns of the same row, and the
            // delete above dropped all of them
            for (size_t k = i; k < batch.size(); ++k) {
              if (batch[k].request_id == batch[i].request_id
                  && batch[k].rowkey == rowkey) {
                const int64_t remaining = count_tasks(table_id, batch[k].column_name);
                share::ai_column_state().mark(
                    atoll(table_id.c_str()), batch[k].column_name.c_str(),
                    static_cast<int64_t>(batch[k].column_name.size()),
                    0 == remaining ? share::ObAIColumnState::READY
                                   : share::ObAIColumnState::PENDING);
              }
            }
          }
          LOG_INFO("ai column worker task dropped", K(tmp2), K(delcnt));
        }
      } else {
        LOG_INFO("ai column worker batch update failed", K(columns_sql.c_str()));
      }
    }
  }

  std::atomic<bool> stop_;
  std::thread thread_;
  bool started_;
};

}  // namespace observer
}  // namespace oceanbase

#endif
