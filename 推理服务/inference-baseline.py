from http.server import BaseHTTPRequestHandler, HTTPServer
import json
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        data=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))))
        result=json.dumps({'prediction':len(data.get('text',''))}).encode()
        self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(result)
if __name__ == '__main__': HTTPServer(('127.0.0.1',8000), Handler).serve_forever()
