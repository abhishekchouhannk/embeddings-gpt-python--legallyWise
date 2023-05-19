# import os
# import http.server
# import socketserver

# from http import HTTPStatus


# class Handler(http.server.SimpleHTTPRequestHandler):
#     def do_GET(self):
#         self.send_response(HTTPStatus.OK)
#         self.end_headers()
#         msg = 'Python is running on Qoddi! You requested %s' % (self.path)
#         self.wfile.write(msg.encode())


# port = int(os.getenv('PORT', 8080))
# print('Listening on port %s' % (port))
# httpd = socketserver.TCPServer(('', port), Handler)
# httpd.serve_forever()

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/hello-world')
def hello_world():
    return 'Hello world'

if __name__ == '__main__':
    app.run(host='localhost', port=8000)