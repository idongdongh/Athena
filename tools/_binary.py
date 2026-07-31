"""二进制扩展名黑名单（工具共享）。

read_file 用它快速拒绝二进制文件；search_files 用它跳过二进制文件。
抽出来避免两份副本漂移——曾经 read_file（38 个）比 search_files（34 个）
多 ``.a .o .svg .wasm``，同一概念两处对不上，导致 .svg 在 read_file 被拒、
在 search_files 却被 grep，模型收到矛盾结论。
"""

BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
    ".pdf", ".zip", ".gz", ".tar", ".tgz", ".bz2", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".class", ".jar", ".pyc", ".o", ".a",
    ".wasm", ".mp3", ".mp4", ".mov", ".avi", ".mkv", ".flac", ".wav",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
}
