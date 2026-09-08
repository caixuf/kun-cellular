#pragma once
#include "tasks/transfer/knowledge_evidence.hpp"
#include <sqlite3.h>
#include <filesystem>

namespace kun::transfer {
struct AdoptionReceipt;

namespace database {
class Statement final {
public:
    Statement(sqlite3* db, const char* sql) : db_(db) {
        require(sqlite3_prepare_v2(db, sql, -1, &stmt_, nullptr) == SQLITE_OK, sqlite3_errmsg(db));
    }
    ~Statement() { sqlite3_finalize(stmt_); }
    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;
    void text(int index, const std::string& value) {
        require(sqlite3_bind_text64(stmt_, index, value.data(), value.size(),
                    SQLITE_TRANSIENT, SQLITE_UTF8) == SQLITE_OK, sqlite3_errmsg(db_));
    }
    void blob(int index, const wire::Bytes& value) {
        require(sqlite3_bind_blob64(stmt_, index, value.data(), value.size(),
                    SQLITE_TRANSIENT) == SQLITE_OK, sqlite3_errmsg(db_));
    }
    void integer(int index, uint64_t value) {
        require(value <= uint64_t(INT64_MAX), "SQLite counter overflow");
        require(sqlite3_bind_int64(stmt_, index, static_cast<sqlite3_int64>(value)) == SQLITE_OK,
                sqlite3_errmsg(db_));
    }
    bool row() {
        const auto rc = sqlite3_step(stmt_);
        require(rc == SQLITE_ROW || rc == SQLITE_DONE, sqlite3_errmsg(db_));
        return rc == SQLITE_ROW;
    }
    void done() { require(!row(), "unexpected database result row"); }
    std::string text(int index) const {
        require(sqlite3_column_type(stmt_, index) == SQLITE_TEXT, "corrupt database: expected text");
        return {reinterpret_cast<const char*>(sqlite3_column_text(stmt_, index)),
                static_cast<std::size_t>(sqlite3_column_bytes(stmt_, index))};
    }
    wire::Bytes blob(int index) const {
        require(sqlite3_column_type(stmt_, index) == SQLITE_BLOB, "corrupt database: expected blob");
        const auto* data = static_cast<const uint8_t*>(sqlite3_column_blob(stmt_, index));
        const auto n = sqlite3_column_bytes(stmt_, index);
        if (!n) return {};
        return {data, data + n};
    }
    uint64_t integer(int index) const {
        require(sqlite3_column_type(stmt_, index) == SQLITE_INTEGER &&
                sqlite3_column_int64(stmt_, index) >= 0, "corrupt database: expected unsigned integer");
        return static_cast<uint64_t>(sqlite3_column_int64(stmt_, index));
    }
private:
    sqlite3* db_;
    sqlite3_stmt* stmt_{nullptr};
};
inline void execute(sqlite3* db, const std::string& sql) {
    char* message = nullptr;
    auto rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &message);
    std::string reason = message ? message : "";
    sqlite3_free(message);
    require(rc == SQLITE_OK, "SQLite: " + reason);
}
class Transaction final {
public:
    explicit Transaction(sqlite3* db, bool write = true) : db_(db) {
        execute(db_, write ? "BEGIN IMMEDIATE" : "BEGIN");
    }
    ~Transaction() { if (!committed_) sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr); }
    void commit() { execute(db_, "COMMIT"); committed_ = true; }
private:
    sqlite3* db_;
    bool committed_{false};
};
inline void ref(Statement& s, const KnowledgeRef& id, int start = 1) {
    s.text(start, id.object_id); s.text(start + 1, id.version);
}
inline const std::vector<std::pair<std::string, std::string>>& schema() {
    static const std::vector<std::pair<std::string, std::string>> sql = [] {
        std::vector<std::pair<std::string, std::string>> v{
            {"knowledge_objects", "CREATE TABLE knowledge_objects(object_id TEXT NOT NULL,version TEXT NOT NULL,title TEXT NOT NULL,artifact BLOB NOT NULL,content_digest TEXT NOT NULL,environment TEXT NOT NULL,interface_id TEXT NOT NULL,producer_lineage TEXT NOT NULL,origin TEXT NOT NULL,PRIMARY KEY(object_id,version))"},
            {"object_status", "CREATE TABLE object_status(object_id TEXT NOT NULL,version TEXT NOT NULL,status TEXT NOT NULL CHECK(status IN ('candidate','research-validated','deprecated','retired')),borrows INTEGER NOT NULL DEFAULT 0 CHECK(typeof(borrows)='integer' AND borrows>=0),failures INTEGER NOT NULL DEFAULT 0 CHECK(typeof(failures)='integer' AND failures>=0),reason TEXT NOT NULL DEFAULT '',PRIMARY KEY(object_id,version),FOREIGN KEY(object_id,version) REFERENCES knowledge_objects(object_id,version))"},
            {"parents", "CREATE TABLE parents(object_id TEXT NOT NULL,version TEXT NOT NULL,parent_id TEXT NOT NULL,parent_version TEXT NOT NULL,PRIMARY KEY(object_id,version,parent_id,parent_version),FOREIGN KEY(object_id,version) REFERENCES knowledge_objects(object_id,version),FOREIGN KEY(parent_id,parent_version) REFERENCES knowledge_objects(object_id,version))"},
            {"evidence", "CREATE TABLE evidence(sequence INTEGER PRIMARY KEY AUTOINCREMENT,object_id TEXT NOT NULL,version TEXT NOT NULL,content_digest TEXT NOT NULL,manifest_digest TEXT NOT NULL,report_digest TEXT NOT NULL,report BLOB NOT NULL,environment TEXT NOT NULL,protocol TEXT NOT NULL,passed INTEGER NOT NULL CHECK(passed IN (0,1)),graph_executions INTEGER NOT NULL,cell_visits INTEGER NOT NULL,edge_visits INTEGER NOT NULL,elapsed_ns INTEGER NOT NULL,FOREIGN KEY(object_id,version) REFERENCES knowledge_objects(object_id,version))"},
            {"events", "CREATE TABLE events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,event TEXT NOT NULL,object_id TEXT NOT NULL,version TEXT NOT NULL,actor TEXT NOT NULL,detail TEXT NOT NULL)"},
            {"adoptions", "CREATE TABLE adoptions(sequence INTEGER PRIMARY KEY AUTOINCREMENT,object_id TEXT NOT NULL,version TEXT NOT NULL,borrow_event INTEGER NOT NULL,organism_id TEXT NOT NULL,boundary_tick TEXT NOT NULL,receipt_digest TEXT NOT NULL,receipt BLOB NOT NULL,UNIQUE(borrow_event,organism_id,boundary_tick),FOREIGN KEY(object_id,version) REFERENCES knowledge_objects(object_id,version),FOREIGN KEY(borrow_event) REFERENCES events(sequence))"},
            {"parents_closed", "CREATE TRIGGER parents_closed BEFORE INSERT ON parents WHEN EXISTS(SELECT 1 FROM object_status WHERE object_id=NEW.object_id AND version=NEW.version) BEGIN SELECT RAISE(ABORT,'publication provenance is sealed'); END"}
        };
        for (const std::string table : {"knowledge_objects", "parents", "evidence", "events", "adoptions"}) {
            const auto prefix = table == "knowledge_objects" ? "objects" : table;
            for (const std::string operation : {"UPDATE", "DELETE"}) {
                const auto name = prefix + (operation == "UPDATE" ? "_no_update" : "_no_delete");
                v.push_back({name, "CREATE TRIGGER " + name + " BEFORE " + operation +
                    " ON " + table + " BEGIN SELECT RAISE(ABORT,'immutable R8 record'); END"});
            }
        }
        return v;
    }();
    return sql;
}
} // namespace database

struct LibraryEntry {
    KnowledgeRef ref;
    std::string title;
    KnowledgeModule module;
    std::vector<KnowledgeRef> parents;
    std::string status;
    uint64_t borrows{0}, failures{0};
};
struct BorrowedKnowledge {
    KnowledgeRef source;
    KnowledgeModule module;
    uint64_t event_sequence{0};
    std::string content_digest;
};
struct LibraryWork {
    uint64_t publication_attempts{0}, publication_validations{0}, read_validations{0};
    uint64_t evaluation_attempts{0}, failed_evaluations{0};
    uint64_t graph_executions{0}, cell_visits{0}, edge_visits{0};
    uint64_t retrieval_attempts{0}, failed_retrievals{0}, artifact_bytes_validated{0};
};

class GermlineLibraryStore final {
public:
    static constexpr int schema_version = 3;
    static constexpr int application_id = 1263881803;
    explicit GermlineLibraryStore(const std::string& path) {
        const bool existed = std::filesystem::exists(path);
        auto rc = sqlite3_open_v2(path.c_str(), &db_,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr);
        if (rc != SQLITE_OK) {
            const std::string reason = db_ ? sqlite3_errmsg(db_) : "SQLite allocation failed";
            if (db_) sqlite3_close(db_);
            db_ = nullptr;
            throw KnowledgeError(reason);
        }
        try {
            require(sqlite3_busy_timeout(db_, 10000) == SQLITE_OK, sqlite3_errmsg(db_));
            database::execute(db_, "PRAGMA foreign_keys=ON");
            database::Transaction tx(db_);
            database::Statement version(db_, "PRAGMA user_version");
            require(version.row(), "missing schema version");
            const auto v = version.integer(0);
            if (v == 0 && !existed) {
                database::Statement objects(db_, "SELECT count(*) FROM sqlite_master");
                require(objects.row() && objects.integer(0) == 0, "refusing to modify unknown schema");
                for (const auto& [name, sql] : database::schema()) database::execute(db_, sql);
                database::execute(db_, "PRAGMA application_id=1263881803; PRAGMA user_version=3");
            } else {
                require(v == schema_version, "unsupported R8 SQLite schema version");
                database::Statement app(db_, "PRAGMA application_id");
                require(app.row() && app.integer(0) == application_id, "wrong SQLite application ID");
                for (const auto& [name, sql] : database::schema()) {
                    database::Statement check(db_, "SELECT sql FROM sqlite_master WHERE name=?");
                    check.text(1, name);
                    require(check.row() && check.text(0) == sql, "R8 schema mismatch: " + name);
                }
            }
            tx.commit();
        } catch (...) { sqlite3_close(db_); db_ = nullptr; throw; }
    }
    ~GermlineLibraryStore() { sqlite3_close(db_); }
    GermlineLibraryStore(const GermlineLibraryStore&) = delete;
    GermlineLibraryStore& operator=(const GermlineLibraryStore&) = delete;

    void publish(const KnowledgeRef& ref, const std::string& title,
                 const KnowledgeModule& module, const std::vector<KnowledgeRef>& parents) {
        ++work_.publication_attempts;
        require(!ref.object_id.empty() && !ref.version.empty() && !title.empty(), "missing publication identity");
        // Fresh decode prevents an artifact from bypassing native validation at the write boundary.
        auto bytes = module.encode();
        auto validated = KnowledgeModule::decode(bytes);
        ++work_.publication_validations;
        work_.artifact_bytes_validated += bytes.size();
        database::Transaction tx(db_);
        for (const auto& parent : parents) load(parent);
        database::Statement insert(db_, "INSERT INTO knowledge_objects VALUES(?,?,?,?,?,?,?,?,?)");
        database::ref(insert, ref); insert.text(3, title); insert.blob(4, bytes);
        insert.text(5, wire::digest(bytes)); insert.text(6, validated.contract().environment);
        insert.text(7, validated.contract().interface_id); insert.text(8, validated.producer_lineage());
        insert.text(9, validated.origin()); insert.done();
        for (const auto& parent : parents) {
            database::Statement p(db_, "INSERT INTO parents VALUES(?,?,?,?)");
            database::ref(p, ref); database::ref(p, parent, 3); p.done();
        }
        database::Statement status(db_, "INSERT INTO object_status(object_id,version,status) VALUES(?,?,'candidate')");
        database::ref(status, ref); status.done();
        event("publish", ref, validated.producer_lineage(), validated.declaration() +
              "; native compile/fresh reconstruction; executions=0; bytes=" + std::to_string(bytes.size()));
        tx.commit();
    }
    LibraryEntry entry(const KnowledgeRef& ref) {
        database::Transaction tx(db_, false);
        auto result = load(ref);
        tx.commit();
        return result;
    }
    std::vector<KnowledgeRef> find(const ModuleContract& contract) {
        database::Transaction tx(db_, false);
        database::Statement select(db_, "SELECT k.object_id,k.version FROM knowledge_objects k JOIN object_status s USING(object_id,version) WHERE s.status='research-validated' ORDER BY k.object_id,k.version");
        std::vector<KnowledgeRef> result;
        while (select.row()) {
            KnowledgeRef ref{select.text(0), select.text(1)};
            auto e = load(ref);
            if (e.module.contract().compatible(contract)) result.push_back(ref);
        }
        tx.commit();
        return result;
    }
    BorrowedKnowledge borrow(const KnowledgeRef& ref, const ModuleContract& contract, const std::string& actor) {
        ++work_.retrieval_attempts;
        try {
            database::Transaction tx(db_);
            auto e = load(ref);
            require(e.status == "research-validated", "knowledge is not available for research borrowing");
            require(e.module.contract().compatible(contract), "borrow interface/environment/timing mismatch");
            database::Statement update(db_, "UPDATE object_status SET borrows=borrows+1 WHERE object_id=? AND version=?");
            database::ref(update, ref); update.done();
            const auto sequence = event("borrow", ref, actor, "research-only; no deployment certification");
            auto content = wire::digest(e.module.encode());
            tx.commit();
            return {ref, std::move(e.module), sequence, std::move(content)};
        } catch (const KnowledgeError& error) {
            ++work_.failed_retrievals;
            failure(ref, actor, "borrow-rejected", error.what());
            throw;
        }
    }
    EvaluationReport evaluate(const KnowledgeRef& ref, const EvaluationProtocol& protocol, const std::string& actor) {
        ++work_.evaluation_attempts;
        try {
            // This intentionally holds the writer transaction through the small native evaluation.
            // Retirement, evidence, eligibility and the audit cannot race each other.
            database::Transaction tx(db_);
            auto e = load(ref);
            auto report = evaluate_native(e.module, protocol);
            work_.graph_executions += report.graph_executions;
            work_.cell_visits += report.cell_visits;
            work_.edge_visits += report.edge_visits;
            work_.failed_evaluations += !report.passed;
            database::Statement insert(db_, "INSERT INTO evidence(object_id,version,content_digest,manifest_digest,report_digest,report,environment,protocol,passed,graph_executions,cell_visits,edge_visits,elapsed_ns) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)");
            database::ref(insert, ref);
            insert.text(3, report.content_digest); insert.text(4, report.manifest_digest);
            insert.text(5, report.report_digest); insert.blob(6, report.bytes);
            insert.text(7, protocol.environment); insert.text(8, protocol.protocol_id);
            insert.integer(9, report.passed); insert.integer(10, report.graph_executions);
            insert.integer(11, report.cell_visits); insert.integer(12, report.edge_visits);
            insert.integer(13, report.elapsed_ns); insert.done();
            database::Statement update(db_, "UPDATE object_status SET status=CASE WHEN status IN ('retired','deprecated') THEN status ELSE ? END,failures=failures+? WHERE object_id=? AND version=?");
            update.text(1, report.passed ? "research-validated" : "candidate");
            update.integer(2, !report.passed); database::ref(update, ref, 3); update.done();
            event(report.passed ? "evaluation-passed" : "evaluation-failed", ref, actor,
                  "sha256:" + report.report_digest + "; graph_executions=" + std::to_string(report.graph_executions));
            tx.commit();
            return report;
        } catch (const KnowledgeError& error) {
            ++work_.failed_evaluations;
            failure(ref, actor, "evaluation-rejected", error.what());
            throw;
        }
    }
    void retire(const KnowledgeRef& ref, const std::string& actor, const std::string& reason,
                bool deprecate = false) {
        require(!reason.empty(), "retirement/deprecation reason required");
        database::Transaction tx(db_);
        auto e = load(ref);
        require(!(deprecate && e.status == "retired"), "retired version cannot be reactivated");
        database::Statement update(db_, "UPDATE object_status SET status=?,reason=? WHERE object_id=? AND version=?");
        update.text(1, deprecate ? "deprecated" : "retired"); update.text(2, reason);
        database::ref(update, ref, 3); update.done();
        event(deprecate ? "deprecate" : "retire", ref, actor, reason);
        tx.commit();
    }
    const LibraryWork& work() const { return work_; }
    void record_adoption(const AdoptionReceipt& receipt, const Phenotype& target);
private:
    LibraryEntry load(const KnowledgeRef& ref) {
        database::Statement s(db_, "SELECT k.title,k.artifact,k.content_digest,k.environment,k.interface_id,k.producer_lineage,k.origin,s.status,s.borrows,s.failures FROM knowledge_objects k JOIN object_status s USING(object_id,version) WHERE k.object_id=? AND k.version=?");
        database::ref(s, ref);
        require(s.row(), "knowledge version not found: " + ref.object_id + "/" + ref.version);
        auto bytes = s.blob(1);
        require(wire::digest(bytes) == s.text(2), "knowledge content digest mismatch");
        auto module = KnowledgeModule::decode(bytes);
        ++work_.read_validations;
        work_.artifact_bytes_validated += bytes.size();
        require(module.contract().environment == s.text(3) && module.contract().interface_id == s.text(4) &&
                module.producer_lineage() == s.text(5) && module.origin() == s.text(6),
                "knowledge metadata/content mismatch");
        LibraryEntry e{ref, s.text(0), std::move(module), {}, s.text(7), s.integer(8), s.integer(9)};
        database::Statement parents(db_, "SELECT parent_id,parent_version FROM parents WHERE object_id=? AND version=? ORDER BY parent_id,parent_version");
        database::ref(parents, ref);
        while (parents.row()) e.parents.push_back({parents.text(0), parents.text(1)});
        database::Statement reports(db_, "SELECT content_digest,report_digest,report,environment,passed,manifest_digest,protocol,graph_executions,cell_visits,edge_visits,elapsed_ns FROM evidence WHERE object_id=? AND version=? ORDER BY sequence");
        database::ref(reports, ref);
        bool latest_pass = false;
        while (reports.row()) {
            const auto report = reports.blob(2);
            require(reports.text(0) == s.text(2) && wire::digest(report) == reports.text(1) &&
                    reports.text(3) == e.module.contract().environment, "evidence/content/environment mismatch");
            auto verified = validate_stored_report(report, e.module);
            require(verified.report.manifest_digest == reports.text(5) &&
                    verified.protocol.protocol_id == reports.text(6) &&
                    verified.report.graph_executions == reports.integer(7) &&
                    verified.report.cell_visits == reports.integer(8) &&
                    verified.report.edge_visits == reports.integer(9) &&
                    verified.report.elapsed_ns == reports.integer(10) &&
                    uint64_t(verified.report.passed) == reports.integer(4), "evidence row/report mismatch");
            latest_pass = verified.report.passed;
        }
        require(e.status != "research-validated" || latest_pass, "eligibility without latest passing native evidence");
        return e;
    }
    uint64_t event(const std::string& kind, const KnowledgeRef& ref,
                   const std::string& actor, const std::string& detail) {
        require(!actor.empty(), "audit actor is required");
        database::Statement s(db_, "INSERT INTO events(event,object_id,version,actor,detail) VALUES(?,?,?,?,?)");
        s.text(1, kind); database::ref(s, ref, 2); s.text(4, actor); s.text(5, detail); s.done();
        return static_cast<uint64_t>(sqlite3_last_insert_rowid(db_));
    }
    void failure(const KnowledgeRef& ref, const std::string& actor, const std::string& kind,
                 const std::string& reason) {
        database::Statement exists(db_,
            "SELECT 1 FROM knowledge_objects WHERE object_id=? AND version=?");
        database::ref(exists, ref);
        if (!exists.row()) return;
        database::Transaction tx(db_);
        database::Statement s(db_, "UPDATE object_status SET failures=failures+1 WHERE object_id=? AND version=?");
        database::ref(s, ref); s.done();
        event(kind, ref, actor, reason);
        tx.commit();
    }
    sqlite3* db_{nullptr};
    LibraryWork work_;
};
} // namespace kun::transfer
