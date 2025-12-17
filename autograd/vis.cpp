
/*******************************
 * Visualization Section (no external libs)
 *
 * Provides:
 *  1) to_dot(...)           : Graphviz DOT text (you can print to console or write to a file)
 *  2) write_png(...)        : writes a PNG file using Graphviz 'dot' command (if installed)
 * Notes:
 *  - Works with shared graphs: tracks visited nodes to avoid infinite recursion.
 *  - Does NOT require any third-party library. DOT is plain text.
 *******************************/

#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <fstream>
#include "vis.h"

/**
 * Helper: build a compact label for a node:
 *   data=...
 *   op=... (or none)
 */
std::string value_label(const std::shared_ptr<Value>& v) {
    std::ostringstream ss;

    if (v->get_label().has_value()) {
        ss << v->get_label().value();
        ss << "\\ndata=" << v->get_data();
    } else {
        ss << "data=" << v->get_data();
    }

    ss << "\\ngrad=" << v->get_grad();

    if (v->get_operation().has_value()) {
        ss << "\\nop=" << to_string(v->get_operation().value());
    }
    return ss.str();
}

/**
 * 2) DOT graph dumper (Graphviz format as plain text)
 *
 * This returns a DOT string describing the computation graph.
 * You can:
 *  - print it to console, or
 *  - write it to a .dot file
 *
 * If you DO have the 'dot' command installed locally, you can render:
 *   dot -Tpng graph.dot -o graph.png
 *
 * But rendering is optional—the DOT text itself is the visualization artifact.
 */
std::string to_dot(const std::shared_ptr<Value>& out) {
    std::ostringstream os;
    os << "digraph autograd {\n";
    os << "  rankdir=LR;\n";
    os << "  node [shape=box];\n";

    // Assign stable numeric IDs per node pointer
    std::unordered_map<const Value*, int> ids;
    std::unordered_set<const Value*> seen; // make sure we only emit each node once
    int next_id = 0;

    std::function<void(const std::shared_ptr<Value>&)> dfs =
        [&](const std::shared_ptr<Value>& v) {
            if (!v) return;

            const Value* ptr = v.get();
            if (!ids.count(ptr)) ids[ptr] = next_id++;

            // emit node once
            if (!seen.count(ptr)) {
                seen.insert(ptr);
                os << "  n" << ids[ptr]
                   << " [label=\"" << value_label(v) << "\"];\n";
            }

            // add edges: prev -> current
            for (const auto& p : v->get_prev()) {
                if (!p) continue;

                const Value* pptr = p.get();
                if (!ids.count(pptr)) ids[pptr] = next_id++;

                os << "  n" << ids[pptr] << " -> n" << ids[ptr] << ";\n";

                // continue traversal
                if (!seen.count(pptr)) dfs(p);
            }
        };

    dfs(out);

    os << "}\n";
    return os.str();
}

/**
 * Convenience: write DOT text to a file.
 * Still no external libs; just uses std::ofstream.
 */
void write_dot_file(const std::shared_ptr<Value>& out, const std::string& path) {
    std::ofstream f(path);
    f << to_dot(out);
    f.close();
}
/**
 * Write a PNG visualization of the computation graph.
 *
 * @param out        Root Value node
 * @param png_path   Output PNG path (e.g. "graph.png")
 * @param dot_path   Temporary DOT path (default: "graph.dot")
 */
void write_png(
    const std::shared_ptr<Value>& out,
    const std::string& png_path,
    const std::string& dot_path
) {
    // 1) Write DOT file
    write_dot_file(out, dot_path);

    // 2) Build Graphviz command
    //    -Tpng : output format
    //    -o    : output file
    std::string cmd = "dot -Tpng " + dot_path + " -o " + png_path;

    // 3) Execute
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        throw std::runtime_error(
            "Graphviz 'dot' command failed. "
            "Is Graphviz installed and on PATH?"
        );
    }
}
