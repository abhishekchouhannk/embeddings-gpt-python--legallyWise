import os
import http.server
import socketserver

PORT = 8000

class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/hello':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Hello from custom route!')
        else:
            super().do_GET()

with socketserver.TCPServer(("", PORT), MyRequestHandler) as httpd:
    print("Server running on port", PORT)
    httpd.serve_forever()
