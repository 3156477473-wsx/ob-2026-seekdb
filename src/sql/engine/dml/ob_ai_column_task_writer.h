#ifndef OB_AI_COLUMN_TASK_WRITER_H
#define OB_AI_COLUMN_TASK_WRITER_H

#include "lib/string/ob_string.h"
#include "lib/allocator/ob_allocator.h"
#include "lib/oblog/ob_log_module.h"
#include "lib/time/ob_time_utility.h"
#include "share/schema/ob_schema_getter_guard.h"
#include "share/schema/ob_table_schema.h"
#include "sql/engine/ob_exec_context.h"
#include "sql/session/ob_sql_session_info.h"
#include "storage/ob_table_dml_param.h"
#include "storage/tx_storage/ob_access_service.h"
#include "storage/ob_value_row_iterator.h"
#include "share/rc/ob_server_runtime.h"
#include "storage/tx/ob_trans_service.h"
#include "share/schema/ob_multi_version_schema_service.h"
#include "sql/das/ob_das_define.h"

namespace oceanbase
{
namespace sql
{
// Writes one row per AI column into oceanbase.__all_ai_column_task inside the
// user transaction, so that the task row is visible to the transaction that
// produced it and disappears when that transaction rolls back.
class ObAIColumnTaskWriter
{
public:
  static int write_tasks_for_insert(ObExecContext &exec_ctx,
                                    const uint64_t table_id,
                                    const ObIArray<uint64_t> &column_ids,
                                    const ObChunkDatumStore::StoredRow &row);
private:
  static bool is_ai_request_column(const ObString &name)
  {
    return name.length() > 9 && 0 == strncmp(name.ptr(), "__ai_req_", 9);
  }

  static int find_task_schema(share::schema::ObSchemaGetterGuard &guard,
                              const ObTableSchema *&task_schema)
  {
    int ret = OB_SUCCESS;
    const share::schema::ObDatabaseSchema *db_schema = NULL;
    task_schema = NULL;
    if (OB_FAIL(guard.get_database_schema(OB_SYS_DATABASE_ID, db_schema))) {
    } else if (OB_ISNULL(db_schema)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(guard.get_table_schema(db_schema->get_database_id(),
                                             ObString::make_string("__all_ai_column_task"),
                                             false, task_schema))) {
    } else if (OB_ISNULL(task_schema)) {
      ret = OB_TABLE_NOT_EXIST;
    }
    return ret;
  }

  static int canonical_table_id(share::schema::ObSchemaGetterGuard &guard,
                                const ObTableSchema &schema,
                                uint64_t &table_id)
  {
    int ret = OB_SUCCESS;
    const ObTableSchema *canonical = NULL;
    if (OB_FAIL(guard.get_table_schema(schema.get_database_id(),
                                       schema.get_table_name_str(),
                                       false, canonical))) {
    } else if (OB_ISNULL(canonical)) {
      ret = OB_TABLE_NOT_EXIST;
    } else {
      table_id = canonical->get_table_id();
    }
    return ret;
  }

  static int build_rowkey_predicate(ObIAllocator &allocator,
                                    const ObTableSchema &table_schema,
                                    const ObIArray<uint64_t> &column_ids,
                                    const ObChunkDatumStore::StoredRow &row,
                                    ObString &predicate)
  {
    int ret = OB_SUCCESS;
    char buffer[512];
    int64_t pos = 0;
    const ObRowkeyInfo &rowkey_info = table_schema.get_rowkey_info();
    for (int64_t i = 0; OB_SUCC(ret) && i < rowkey_info.get_size(); ++i) {
      uint64_t rowkey_cid = OB_INVALID_ID;
      int64_t cell = -1;
      if (OB_FAIL(rowkey_info.get_column_id(i, rowkey_cid))) {
      } else {
        for (int64_t idx = 0; idx < column_ids.count(); ++idx) {
          if (column_ids.at(idx) == rowkey_cid) {
            cell = idx;
            break;
          }
        }
      }
      if (OB_FAIL(ret)) {
      } else if (cell < 0 || cell >= row.cnt_) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        const ObDatum &datum = row.cells()[cell];
        const ObColumnSchemaV2 *column = table_schema.get_column_schema(rowkey_cid);
        if (OB_ISNULL(column)) {
          ret = OB_ERR_UNEXPECTED;
        } else if (i > 0 && OB_FAIL(databuff_printf(buffer, sizeof(buffer), pos, " AND "))) {
        } else if (OB_FAIL(databuff_printf(buffer, sizeof(buffer), pos, "`%.*s` = ",
                                           column->get_column_name_str().length(),
                                           column->get_column_name_str().ptr()))) {
        } else if (ob_is_integer_type(column->get_data_type())
                   || ObNumberType == column->get_data_type()
                   || ObDecimalIntType == column->get_data_type()) {
          if (OB_FAIL(databuff_printf(buffer, sizeof(buffer), pos, "%ld", datum.get_int()))) {
          }
        } else {
          if (OB_FAIL(databuff_printf(buffer, sizeof(buffer), pos, "%.*s",
                                      static_cast<int>(datum.len_), datum.ptr_))) {
          }
        }
      }
    }
    if (OB_SUCC(ret)) {
      if (OB_FAIL(ob_write_string(allocator, ObString(pos, buffer), predicate))) {
      }
    }
    return ret;
  }

  static int write_one_row(const uint64_t table_id,
                           const ObTableSchema &table_schema,
                           const ObColumnSchemaV2 &ai_column,
                           const int64_t request_id,
                           const ObString &predicate,
                           transaction::ObTxDesc &tx_desc,
                           storage::ObAccessService &access_service)
  {
    int ret = OB_SUCCESS;
    ObArenaAllocator allocator("AIColTask");
    ObSEArray<uint64_t, 16> all_column_ids;
    share::schema::ObTableDMLParam table_param(allocator);
    ObDMLBaseParam dml_param;
    transaction::ObTxReadSnapshot snapshot;
    storage::ObStoreCtxGuard store_ctx_guard;
    storage::ObValueRowIterator row_iter;
    ObArenaAllocator datum_allocator("AIColDatum");

    transaction::ObTransService *txs = NULL;
    if (OB_ISNULL(txs = share::server_service<transaction::ObTransService>())) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(txs->get_read_snapshot(tx_desc,
                                              transaction::ObTxIsolationLevel::RC,
                                              THIS_WORKER.get_timeout_ts(),
                                              snapshot))) {
      LOG_INFO("ai task get snapshot failed", K(ret));
    } else if (OB_FAIL(ai_column_table_all_ids(table_schema, all_column_ids))) {
    } else if (OB_FAIL(table_param.convert(&table_schema, table_schema.get_schema_version(),
                                           all_column_ids))) {
      LOG_INFO("ai task convert failed", K(ret), K(all_column_ids.count()));
    } else if (OB_FAIL(access_service.get_write_store_ctx_guard(
                 THIS_WORKER.get_timeout_ts(), tx_desc, snapshot, 0,
                 dml_param.write_flag_, store_ctx_guard))) {
      LOG_INFO("ai task store ctx failed", K(ret));
    } else if (OB_FAIL(dml_param.snapshot_.assign(snapshot))) {
    } else {
      dml_param.sql_mode_ = SMO_DEFAULT;
      dml_param.write_flag_.reset();
      dml_param.write_flag_.set_is_insert_up();
      dml_param.table_param_ = &table_param;
      dml_param.timeout_ = THIS_WORKER.get_timeout_ts();
      dml_param.branch_id_ = 0;
      dml_param.store_ctx_guard_ = &store_ctx_guard;
      dml_param.schema_version_ = table_schema.get_schema_version();
      dml_param.dml_allocator_ = &allocator;
      blocksstable::ObDatumRow datum_row;
      if (OB_FAIL(datum_row.init(all_column_ids.count()))) {
        LOG_INFO("init datum row failed", K(ret));
      } else {
        ObString short_name;
        const ObString ai_name = ai_column.get_column_name_str();
        if (OB_FAIL(ob_write_string(allocator, ai_name, short_name))) {
        } else {
          const ObString hidden_name = ai_column.get_column_name_str();
          const ObString base_name = hidden_name.length() > 9
                                         ? ObString(hidden_name.length() - 9, hidden_name.ptr() + 9)
                                         : hidden_name;
          for (int64_t i = 0; OB_SUCC(ret) && i < all_column_ids.count(); ++i) {
            const ObColumnSchemaV2 *column = table_schema.get_column_schema(all_column_ids.at(i));
            ObDatum &datum = datum_row.storage_datums_[i];
            if (OB_ISNULL(column)) {
              ret = OB_ERR_UNEXPECTED;
            } else if (column->get_column_name_str().length() >= 4
                       && 0 == strncmp(column->get_column_name_str().ptr(), "gmt_", 4)) {
              datum.set_datetime(ObTimeUtility::current_time());
            } else if (0 == column->get_column_name_str().case_compare("table_id")) {
              datum.set_int(static_cast<int64_t>(table_id));
            } else if (0 == column->get_column_name_str().case_compare("column_id")) {
              datum.set_int(static_cast<int64_t>(ai_column.get_column_id()));
            } else if (0 == column->get_column_name_str().case_compare("rowkey_predicate")) {
              datum.set_string(predicate);
            } else if (0 == column->get_column_name_str().case_compare("request_id")) {
              datum.set_int(request_id);
            } else if (0 == column->get_column_name_str().case_compare("state")) {
              datum.set_int(0);
            } else if (0 == column->get_column_name_str().case_compare("retry_count")) {
              datum.set_int(0);
            } else if (0 == column->get_column_name_str().case_compare("eligible_ts")) {
              datum.set_datetime(ObTimeUtility::current_time());
            } else if (0 == column->get_column_name_str().case_compare("column_name")) {
              datum.set_string(base_name);
            } else {
              datum.set_null();
            }
          }
          if (OB_FAIL(ret)) {
          } else if (OB_FAIL(row_iter.init())) {
          } else if (OB_FAIL(row_iter.add_row(datum_row))) {
          } else {
            int64_t affected_rows = 0;
            LOG_INFO("ai task before insert", K(table_id), K(all_column_ids.count()),
                     "tablet", table_schema.get_tablet_id().id());
            if (OB_FAIL(access_service.insert_rows(table_schema.get_tablet_id(), tx_desc, dml_param,
                                                   all_column_ids, &row_iter, affected_rows))) {
              LOG_INFO("insert ai column task row failed", K(ret), K(table_id));
              ret = OB_SUCCESS;
            } else {
              LOG_INFO("insert ai column task row", K(table_id),
                       "column_id", ai_column.get_column_id(), "affected", affected_rows);
            }
          }
        }
      }
    }
    row_iter.reset();
    return ret;
  }

  static int ai_column_table_all_ids(const ObTableSchema &table_schema,
                                     ObIArray<uint64_t> &ids)
  {
    int ret = OB_SUCCESS;
    ObSEArray<uint64_t, 16> all_ids;
    if (OB_FAIL(table_schema.get_all_column_ids(all_ids))) {
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < all_ids.count(); ++i) {
      const ObColumnSchemaV2 *column = table_schema.get_column_schema(all_ids.at(i));
      if (OB_ISNULL(column)) {
        ret = OB_ERR_UNEXPECTED;
      } else if (OB_FAIL(ids.push_back(all_ids.at(i)))) {
      }
    }
    return ret;
  }


  static bool expr_mentions_ident(const ObString &expr, const ObString &name)
  {
    bool found = false;
    if (expr.length() > 0 && name.length() > 0) {
      const char *base = expr.ptr();
      const int64_t total = expr.length();
      const int64_t len = name.length();
      for (int64_t i = 0; !found && i + len <= total; ++i) {
        if (0 != strncasecmp(base + i, name.ptr(), static_cast<size_t>(len))) {
          continue;
        }
        const bool left_ok = (0 == i) || !is_ident_char(base[i - 1]);
        const bool right_ok = (i + len >= total) || !is_ident_char(base[i + len]);
        found = left_ok && right_ok;
      }
    }
    return found;
  }

  static bool is_ident_char(const char c)
  {
    return ('_' == c) || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
           || (c >= '0' && c <= '9');
  }

  static bool dependency_assigned(const ObTableSchema &table_schema,
                                  const ObIArray<ObString> &assigned_names,
                                  const ObString &expr)
  {
    bool assigned = false;
    for (int64_t i = 0; !assigned && i < assigned_names.count(); ++i) {
      const ObString &name = assigned_names.at(i);
      if (name.empty()) {
        continue;
      }
      assigned = expr_mentions_ident(expr, name);
      if (!assigned) {
        const ObColumnSchemaV2 *col = table_schema.get_column_schema(name);
        assigned = NULL != col && col->is_rowkey_column();
      }
    }
    return assigned;
  }


  static bool dependency_changed(const ObTableSchema &table_schema,
                                 const ObIArray<uint64_t> &column_ids,
                                 const ObChunkDatumStore::StoredRow &new_row,
                                 const ObChunkDatumStore::StoredRow *old_row,
                                 const ObString &expr)
  {
    bool changed = false;
    if (NULL == old_row) {
      changed = true;
    } else {
      const ObDatum *new_cells = new_row.cells();
      const ObDatum *old_cells = old_row->cells();
      for (int64_t i = 0; !changed && i < column_ids.count(); ++i) {
        const ObColumnSchemaV2 *col = table_schema.get_column_schema(column_ids.at(i));
        if (NULL == col || col->is_hidden()) {
          continue;
        }
        if (!expr_mentions_ident(expr, col->get_column_name_str())) {
          continue;
        }
        if (i >= new_row.cnt_ || i >= old_row->cnt_ || NULL == new_cells || NULL == old_cells) {
          continue;
        }
        if (!common::ObDatum::binary_equal(new_cells[i], old_cells[i])) {
          changed = true;
        }
      }
    }
    return changed;
  }


  static bool update_dependency_changed(const ObTableSchema &table_schema,
                                        const ObIArray<ObString> &assigned_names,
                                        const ObColumnSchemaV2 &req_column)
  {
    bool changed = true;
    ObString expr;
    const ObString cmt = req_column.get_comment_str();
    const char *mark = "ai:expr=";
    const int64_t mark_len = 8;
    if (cmt.length() > mark_len) {
      for (int64_t i = 0; i + mark_len <= cmt.length(); ++i) {
        if (0 == strncmp(cmt.ptr() + i, mark, mark_len)) {
          expr.assign_ptr(cmt.ptr() + i + mark_len, static_cast<int32_t>(cmt.length() - i - mark_len));
          break;
        }
      }
    }
    if (expr.length() > 0) {
      changed = dependency_assigned(table_schema, assigned_names, expr);
    }
    return changed;
  }

public:
  // Builds the inner SQL that removes the tasks of one deleted row. It runs
  // through the DML operator's inner SQL connection so that it shares the user
  // transaction, the same way a foreign key cascade does.
  static int build_delete_task_sql(ObIAllocator &allocator,
                                   const uint64_t table_id,
                                   const ObIArray<uint64_t> &column_ids,
                                   const ObChunkDatumStore::StoredRow &row,
                                   ObString &sql)
  {
    int ret = OB_SUCCESS;
    uint64_t user_table_id = table_id;
    share::schema::ObSchemaGetterGuard guard;
    const ObTableSchema *table_schema = NULL;
    const ObTableSchema *task_schema = NULL;
    ObString predicate;
    const int64_t buf_len = 1024;
    char *buf = NULL;
    int64_t pos = 0;
    bool has_ai_column = false;
    sql.reset();
    if (OB_ISNULL(buf = static_cast<char *>(allocator.alloc(buf_len)))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (OB_FAIL(share::schema::ObMultiVersionSchemaService::get_instance()
                         .get_runtime_schema_guard(guard))) {
    } else if (OB_FAIL(guard.get_table_schema(table_id, table_schema)) || OB_ISNULL(table_schema)) {
    } else if (OB_FAIL(canonical_table_id(guard, *table_schema, user_table_id))) {
    } else if (OB_FAIL(guard.get_table_schema(user_table_id, table_schema)) || OB_ISNULL(table_schema)) {
    }
    for (ObTableSchema::const_column_iterator iter = table_schema->column_begin();
         OB_SUCC(ret) && NULL != table_schema && !has_ai_column
             && iter != table_schema->column_end();
         ++iter) {
      const ObColumnSchemaV2 *column = *iter;
      if (NULL != column && column->is_hidden()
          && is_ai_request_column(column->get_column_name_str())) {
        has_ai_column = true;
      }
    }
    if (OB_FAIL(ret) || !has_ai_column) {
      // no ai column on this table, nothing to remove
    } else if (OB_FAIL(find_task_schema(guard, task_schema)) || OB_ISNULL(task_schema)) {
    } else if (OB_FAIL(build_rowkey_predicate(allocator, *table_schema, column_ids, row, predicate))) {
    } else if (OB_FAIL(databuff_printf(buf, buf_len, pos,
                 "delete from oceanbase.`%.*s` where table_id = %lu and rowkey_predicate = '",
                 static_cast<int>(task_schema->get_table_name_str().length()),
                 task_schema->get_table_name_str().ptr(),
                 user_table_id))) {
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < predicate.length(); ++i) {
      const char c = predicate.ptr()[i];
      if ('\'' == c || '\\' == c) {
        if (OB_FAIL(databuff_printf(buf, buf_len, pos, "\\%c", c))) {
        }
      } else if (OB_FAIL(databuff_printf(buf, buf_len, pos, "%c", c))) {
      }
    }
    if (OB_SUCC(ret) && pos > 0) {
      if (OB_FAIL(databuff_printf(buf, buf_len, pos, "'"))) {
      } else if (OB_FAIL(ob_write_string(allocator, ObString(static_cast<int32_t>(pos), buf), sql))) {
      }
    }
    return ret;
  }

  static int remove_tasks_for_deleted_row(ObExecContext &exec_ctx,
                                          const uint64_t table_id,
                                          const ObIArray<uint64_t> &column_ids,
                                          const ObChunkDatumStore::StoredRow &row)
  {
    int ret = OB_SUCCESS;
    uint64_t user_table_id = table_id;
    ObSQLSessionInfo *session = exec_ctx.get_my_session();
    transaction::ObTxDesc *tx_desc = NULL;
    storage::ObAccessService *access_service = NULL;
    transaction::ObTransService *txs = NULL;
    share::schema::ObSchemaGetterGuard guard;
    const ObTableSchema *table_schema = NULL;
    const ObTableSchema *task_schema = NULL;
    ObArenaAllocator allocator("aiTaskDel");
    ObTxReadSnapshot snapshot;
    
    if (OB_ISNULL(session) || OB_ISNULL(access_service = share::server_service<storage::ObAccessService>())
        || OB_ISNULL(txs = share::server_service<transaction::ObTransService>())) {
      ret = OB_SUCCESS;
    } else if (OB_ISNULL(tx_desc = session->get_tx_desc())) {
      ret = OB_SUCCESS;
    } else if (OB_FAIL(share::schema::ObMultiVersionSchemaService::get_instance()
                         .get_runtime_schema_guard(guard))) {
      LOG_INFO("ai task delete guard failed", K(ret));
      ret = OB_SUCCESS;
    } else if (OB_FAIL(guard.get_table_schema(table_id, table_schema))
               || OB_ISNULL(table_schema)) {
      ret = OB_SUCCESS;
    } else if (OB_FAIL(canonical_table_id(guard, *table_schema, user_table_id))) {
      ret = OB_SUCCESS;
    } else if (OB_FAIL(guard.get_table_schema(user_table_id, table_schema))
               || OB_ISNULL(table_schema) || !table_schema->is_user_table()) {
      ret = OB_SUCCESS;
    } else if (OB_FAIL(find_task_schema(guard, task_schema)) || OB_ISNULL(task_schema)) {
      ret = OB_SUCCESS;
    } else if (OB_FAIL(txs->get_read_snapshot(*tx_desc, transaction::ObTxIsolationLevel::RC,
                                              THIS_WORKER.get_timeout_ts(), snapshot))) {
      LOG_INFO("ai task delete snapshot failed", K(ret));
      ret = OB_SUCCESS;
    } else {
      ObString predicate;
      ObSEArray<uint64_t, 16> all_ids;
      const ObIArray<uint64_t> *pred_ids = &column_ids;
      if (OB_FAIL(table_schema->get_all_column_ids(all_ids))) {
        pred_ids = &column_ids;
      } else if (row.cnt_ == all_ids.count()) {
        pred_ids = &all_ids;
      }
      
      if (OB_FAIL(build_rowkey_predicate(allocator, *table_schema, *pred_ids, row, predicate))) {
        LOG_INFO("ai task delete predicate failed", K(ret));
        ret = OB_SUCCESS;
      } else {
        
        for (ObTableSchema::const_column_iterator iter = table_schema->column_begin();
             OB_SUCC(ret) && iter != table_schema->column_end(); ++iter) {
          const ObColumnSchemaV2 *column = *iter;
          if (NULL == column || !column->is_hidden() || !is_ai_request_column(column->get_column_name_str())) {
            continue;
          }
          
          ObSEArray<uint64_t, 4> key_ids;
          ObDatum key_datums[3];
          blocksstable::ObDatumRow key_row;
          ObTableDMLParam table_param(allocator);
          ObDMLBaseParam dml_param;
          ObStoreCtxGuard guard_ctx;
          ObValueRowIterator row_iter;
          int64_t affected = 0;
          key_datums[0].set_uint(table_schema->get_table_id());
          key_datums[1].set_uint(column->get_column_id());
          key_datums[2].set_string(predicate);
          const ObColumnSchemaV2 *c_table = task_schema->get_column_schema(ObString::make_string("table_id"));
          const ObColumnSchemaV2 *c_column = task_schema->get_column_schema(ObString::make_string("column_id"));
          const ObColumnSchemaV2 *c_pred = task_schema->get_column_schema(ObString::make_string("rowkey_predicate"));
          
          if (OB_ISNULL(c_table) || OB_ISNULL(c_column) || OB_ISNULL(c_pred)) {
            ret = OB_SUCCESS;
          } else if (OB_FAIL(key_ids.push_back(c_table->get_column_id()))) {
          } else if (OB_FAIL(key_ids.push_back(c_column->get_column_id()))) {
          } else if (OB_FAIL(key_ids.push_back(c_pred->get_column_id()))) {
          } else if (OB_FAIL(key_row.init(key_ids.count()))) {
          } else if (FALSE_IT(MEMCPY(key_row.storage_datums_, key_datums, sizeof(key_datums)))) {
          } else if (OB_FAIL(table_param.convert(task_schema, task_schema->get_schema_version(), key_ids))) {
          } else if (FALSE_IT(dml_param.sql_mode_ = session->get_sql_mode())) {
          } else if (FALSE_IT(dml_param.table_param_ = &table_param)) {
          } else if (FALSE_IT(dml_param.timeout_ = THIS_WORKER.get_timeout_ts())) {
          } else if (FALSE_IT(dml_param.snapshot_.assign(snapshot))) {
          } else if (FALSE_IT(dml_param.schema_version_ = task_schema->get_schema_version())) {
          } else if (FALSE_IT(dml_param.dml_allocator_ = &allocator)) {
          } else if (OB_FAIL(access_service->get_write_store_ctx_guard(
                       THIS_WORKER.get_timeout_ts(), *tx_desc, snapshot, 0,
                       dml_param.write_flag_, guard_ctx))) {
            LOG_INFO("ai task delete guard ctx failed", K(ret));
            ret = OB_SUCCESS;
          } else if (FALSE_IT(dml_param.store_ctx_guard_ = &guard_ctx)) {
          } else if (OB_FAIL(row_iter.init())) {
          } else if (OB_FAIL(row_iter.add_row(key_row))) {
          } else if (OB_FALSE_IT(dml_param.skip_old_row_check_ = true)) {
          } else if (OB_FAIL(access_service->delete_rows(task_schema->get_tablet_id(), *tx_desc,
                                                         dml_param, key_ids, &row_iter, affected))) {
            LOG_INFO("ai task delete rows failed", K(ret), K(affected));
            ret = OB_SUCCESS;
          } else {
            LOG_INFO("ai task rows removed", K(affected));
          }
        }
      }
    }
    if (OB_SUCC(ret)) {
      // keep DML path unaffected
    }
    return OB_SUCCESS;
  }

  static int write_tasks_for_update(ObExecContext &exec_ctx,
                                    const uint64_t table_id,
                                    const ObIArray<uint64_t> &column_ids,
                                    const ObChunkDatumStore::StoredRow *row,
                                    const ObIArray<ObString> &assigned_names)
  {
    int ret = common::OB_SUCCESS;
    if (OB_ISNULL(row)) {
      LOG_INFO("ai column task writer skip update: row is null", K(table_id));
    } else {
      ret = write_tasks_for_insert_impl(exec_ctx, table_id, column_ids, *row,
                                        true, &assigned_names);
    }
    return ret;
  }

  static int write_tasks_for_insert_impl(ObExecContext &exec_ctx,
                                         const uint64_t table_id,
                                         const ObIArray<uint64_t> &column_ids,
                                         const ObChunkDatumStore::StoredRow &row,
                                         const bool filter_by_update = false,
                                         const ObIArray<ObString> *assigned_names = NULL)
  {
    int ret = OB_SUCCESS;
    uint64_t user_table_id = OB_INVALID_ID;
    ObSQLSessionInfo *session = exec_ctx.get_my_session();
    transaction::ObTxDesc *tx_desc = NULL;
    storage::ObAccessService *access_service = NULL;
    share::schema::ObSchemaGetterGuard guard;
    const ObTableSchema *table_schema = NULL;
    const ObTableSchema *task_schema = NULL;
    ObArenaAllocator allocator("AIColTaskMgr");
    if (OB_ISNULL(session) || OB_ISNULL(tx_desc = session->get_tx_desc())) {
      ret = OB_INVALID_ARGUMENT;
    } else if (OB_ISNULL(access_service =
                 share::server_service<storage::ObAccessService>())) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(share::schema::ObMultiVersionSchemaService::get_instance()
                          .get_runtime_schema_guard(guard))) {
      LOG_INFO("ai task guard init failed", K(ret));
    } else if (OB_FAIL(guard.get_table_schema(table_id, table_schema))) {
      LOG_INFO("ai task table schema failed", K(ret), K(table_id));
    } else if (OB_ISNULL(table_schema) || !table_schema->is_user_table()) {
      ret = OB_TABLE_NOT_EXIST;
    } else if (OB_FAIL(canonical_table_id(guard, *table_schema, user_table_id))) {
      LOG_INFO("ai task canonical id failed", K(ret), K(table_id));
    } else if (OB_FAIL(find_task_schema(guard, task_schema))) {
    } else {
      LOG_INFO("ai task ids", K(table_id), K(user_table_id),
               "db_id", table_schema->get_database_id(),
               "name", table_schema->get_table_name_str(),
               "schema_table_id", table_schema->get_table_id(),
               "has_partition", table_schema->has_partition(),
               "schema_version", table_schema->get_schema_version());
      const int64_t row_request_id = ObTimeUtility::current_time();
      for (int64_t i = 0; OB_SUCC(ret) && i < table_schema->get_column_count(); ++i) {
        const ObColumnSchemaV2 *column = table_schema->get_column_schema_by_idx(i);
        if (OB_ISNULL(column)) {
          ret = OB_ERR_UNEXPECTED;
        } else if (!is_ai_request_column(column->get_column_name_str())) {
          // not an AI column
        } else {
          ObString predicate;
          if (OB_FAIL(build_rowkey_predicate(allocator, *table_schema, column_ids, row, predicate))) {
            LOG_INFO("ai task predicate failed", K(ret), K(table_id),
                     "column_id", column->get_column_id());
          } else if (filter_by_update
                     && (NULL == assigned_names
                         || !update_dependency_changed(*table_schema, *assigned_names,
                                                       *column))) {
            // the columns referenced by this AI expression were not modified
          } else if (OB_FAIL(write_one_row(user_table_id, *task_schema, *column, row_request_id,
                                           predicate, *tx_desc, *access_service))) {
            LOG_INFO("ai task write one row failed", K(ret), K(table_id),
                     "column_id", column->get_column_id());
          }
        }
      }
    }
    if (OB_FAIL(ret)) {
      LOG_INFO("write ai column tasks failed", K(ret), K(table_id));
      ret = OB_SUCCESS;
    }
    return ret;
  }
};
} // namespace sql
} // namespace oceanbase
#endif