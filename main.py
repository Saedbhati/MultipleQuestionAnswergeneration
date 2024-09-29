from flask import Flask, request, render_template, jsonify
from Model import Model

app = Flask(__name__)

# Initialize model
model = Model()

@app.route('/genrate', methods=['POST'])
def Genrate():
    # Extract data from form request
    context = request.form.get('context', '')
    num_QA = request.form.get('num_QA', '1')
    
    # Convert num_QA to integer
    try:
        num_QA = int(num_QA)
    except ValueError:
        return jsonify({'error': 'Invalid num_QA value'}), 400
    
    # Ensure context is provided
    if not context:
        return jsonify({'error': 'Context is required'}), 400
    
    # Call the model's generate method
    data = model.gen(context, num_QA)
    
    # Return the generated data as a JSON response
    return jsonify(data), 200

@app.route('/', methods=['GET'])
def home():
    # Render the HTML template for the front end
    return render_template('index.html')

if __name__ == "__main__":
    app.run('0.0.0.0', 8080)
