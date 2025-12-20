
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
#include "autograd.h"
#pragma once

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
std::string to_dot(const std::shared_ptr<Value>& out);
/**
 * Convenience: write DOT text to a file.
 * Still no external libs; just uses std::ofstream.
 */
void write_dot_file(const std::shared_ptr<Value>& out, const std::string& path);
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
    const std::string& dot_path = "graph.dot"
);