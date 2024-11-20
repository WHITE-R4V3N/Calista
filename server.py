# Author:       Emma Gillespie
# Date:         2024-11-20
# Description:  A basic flask webserver to host Calista.

import subprocess

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/run', method=['POST'])
def run_command():
    user_input = request.json.get('input', '')
    result = subprocess.run(['python', 'calista.py', user_input], capture_output=True, text=True)
    
    return jsonify({'output': result.stdout})

if __name__ == '__main__':
    app.run(debug=True)