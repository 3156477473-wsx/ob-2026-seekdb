/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OCEANBASE_SQL_PARSER_OB_PARSER_AI_RETRY_H_
#define OCEANBASE_SQL_PARSER_OB_PARSER_AI_RETRY_H_

#include <stdint.h>

namespace oceanbase
{
namespace sql
{
// An AI column retry statement is rewritten into a plain update by the parser,
// so the parser pushes what it saw into this per thread marker and the resolver
// picks it up. The parser and the resolver of one statement always run in the
// same thread, and the marker is reset at the beginning of every parse.
// 0: not an AI column retry, 1: retry one AI column, 2: retry mixed with another
// alter action.
extern thread_local int64_t g_ai_retry_kind;

} // namespace sql
} // namespace oceanbase

#endif // OCEANBASE_SQL_PARSER_OB_PARSER_AI_RETRY_H_
