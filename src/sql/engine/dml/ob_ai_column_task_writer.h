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
      int64_t row_idx = -1;
      if (OB_FAIL(rowkey_info.get_column_id(i, rowkey_cid))) {
      } else {
        for (int64_t idx = 0; idx < table_schema.get_column_count(); ++idx) {
          const ObColumnSchemaV2 *column = table_schema.get_column_schema_by_idx(idx);
          if (OB_NOT_NULL(column) && column->get_column_id() == rowkey_cid) {
            row_idx = idx;
            break;
          }
        }
      }
      if (OB_FAIL(ret)) {
      } else if (row_idx < 0 || row_idx >= row.cnt_) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        const ObDatum &datum = row.cells()[row_idx];
        if (i > 0 && OB_FAIL(databuff_printf(buffer, sizeof(buffer), pos, " AND "))) {
        }
        if (OB_FAIL(ret)) {
        } else if (OB_FAIL(databuff_printf(buffer, sizeof(buffer), pos, "`%s` = ",
                                           table_schema.get_column_schema_by_idx(row_idx)->get_column_name()))) {
        } else if (OB_FAIL(databuff_printf(buffer, sizeof(buffer), pos, "%.*s",
                                           static_cast<int>(datum.len_), datum.ptr_))) {
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

    if (OB_FAIL(ai_column_table_all_ids(table_schema, all_column_ids))) {
    } else if (OB_FAIL(table_param.convert(&table_schema, table_schema.get_schema_version(),
                                           all_column_ids))) {
    } else if (OB_FAIL(access_service.get_write_store_ctx_guard(
                 THIS_WORKER.get_timeout_ts(), tx_desc, snapshot, 0,
                 dml_param.write_flag_, store_ctx_guard))) {
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
          for (int64_t i = 0; OB_SUCC(ret) && i < all_column_ids.count(); ++i) {
            const ObColumnSchemaV2 *column = table_schema.get_column_schema(all_column_ids.at(i));
            ObDatum &datum = datum_row.storage_datums_[i];
            if (OB_ISNULL(column)) {
              ret = OB_ERR_UNEXPECTED;
            } else if (0 == column->get_column_name_str().case_compare("table_id")) {
              datum.set_int(static_cast<int64_t>(table_id));
            } else if (0 == column->get_column_name_str().case_compare("column_id")) {
              datum.set_int(static_cast<int64_t>(ai_column.get_column_id()));
            } else if (0 == column->get_column_name_str().case_compare("rowkey_predicate")) {
              datum.set_string(predicate);
            } else if (0 == column->get_column_name_str().case_compare("request_id")) {
              datum.set_int(ObTimeUtility::current_time());
            } else if (0 == column->get_column_name_str().case_compare("state")) {
              datum.set_int(0);
            } else if (0 == column->get_column_name_str().case_compare("retry_count")) {
              datum.set_int(0);
            } else if (0 == column->get_column_name_str().case_compare("eligible_ts")) {
              datum.set_datetime(ObTimeUtility::current_time());
            } else if (0 == column->get_column_name_str().case_compare("column_name")) {
              datum.set_string(short_name);
            } else {
              datum.set_null();
            }
          }
          if (OB_FAIL(ret)) {
          } else if (OB_FAIL(row_iter.init())) {
          } else if (OB_FAIL(row_iter.add_row(datum_row))) {
          } else {
            int64_t affected_rows = 0;
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
    return table_schema.get_all_column_ids(ids);
  }

public:
  static int write_tasks_for_insert_impl(ObExecContext &exec_ctx,
                                         const uint64_t table_id,
                                         const ObIArray<uint64_t> &column_ids,
                                         const ObChunkDatumStore::StoredRow &row)
  {
    int ret = OB_SUCCESS;
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
    } else if (OB_FAIL(guard.get_table_schema(table_id, table_schema))) {
    } else if (OB_ISNULL(table_schema) || !table_schema->is_user_table()) {
      ret = OB_TABLE_NOT_EXIST;
    } else if (OB_FAIL(find_task_schema(guard, task_schema))) {
    } else {
      for (int64_t i = 0; OB_SUCC(ret) && i < column_ids.count(); ++i) {
        const ObColumnSchemaV2 *column = table_schema->get_column_schema(column_ids.at(i));
        if (OB_ISNULL(column)) {
          ret = OB_ERR_UNEXPECTED;
        } else if (!is_ai_request_column(column->get_column_name_str())) {
          // not an AI column
        } else {
          ObString predicate;
          if (OB_FAIL(build_rowkey_predicate(allocator, *table_schema, column_ids, row, predicate))) {
          } else if (OB_FAIL(write_one_row(table_id, *task_schema, *column, predicate,
                                           *tx_desc, *access_service))) {
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