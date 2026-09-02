#!/usr/bin/env python3
import os
import sys
import http.server
import socketserver
import webbrowser

PORT = 8833
DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def run():
    os.chdir(DIRECTORY)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("======================================================================")
        print(" 硅基细胞计算机全息观测台 (Silicon Cellular Computer Observatory)")
        print(f" 服务已启动: http://localhost:{PORT}/cellular.html")
        print(f" 具身迷宫沙盒: http://localhost:{PORT}/maze.html")
        print(f" 静态根目录: {DIRECTORY}")
        print("======================================================================")
        print("按 Ctrl+C 退出服务...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务已停止。")

if __name__ == "__main__":
    run()
