import numpy as np
import pickle
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('medical.pkl', 'rb'))

# HTML template with Tailwind CSS for a modern, attractive UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Insurance Cost Predictor</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gradient-to-br from-blue-50 via-indigo-50 to-slate-100 min-h-screen flex flex-col justify-between py-10 px-4">

    <div class="max-w-xl w-full mx-auto bg-white shadow-xl rounded-2xl overflow-hidden border border-gray-100">
        <!-- Header -->
        <div class="bg-gradient-to-r from-blue-600 to-indigo-600 py-6 px-8 text-center text-white">
            <h1 class="text-2xl font-bold tracking-tight">Medical Insurance Cost Predictor</h1>
            <p class="text-blue-100 text-sm mt-1">Enter your details below to estimate your medical charges</p>
        </div>

        <!-- Form Body -->
        <form action="/predict" method="POST" class="p-8 space-y-5">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
                <!-- Age -->
                <div>
                    <label class="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1">Age</label>
                    <input type="number" name="age" required min="1" max="120" placeholder="e.g., 30"
                           class="w-full px-4 py-2.5 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:bg-white focus:outline-none transition text-sm">
                </div>

                <!-- Sex -->
                <div>
                    <label class="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1">Sex</label>
                    <select name="sex" class="w-full px-4 py-2.5 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:bg-white focus:outline-none transition text-sm">
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>

                <!-- BMI -->
                <div>
                    <label class="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1">BMI</label>
                    <input type="number" step="0.1" name="bmi" required placeholder="e.g., 24.5"
                           class="w-full px-4 py-2.5 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:bg-white focus:outline-none transition text-sm">
                </div>

                <!-- Children -->
                <div>
                    <label class="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1">Children</label>
                    <input type="number" name="children" required min="0" max="10" placeholder="e.g., 2"
                           class="w-full px-4 py-2.5 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:bg-white focus:outline-none transition text-sm">
                </div>

                <!-- Smoker -->
                <div>
                    <label class="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1">Smoker</label>
                    <select name="smoker" class="w-full px-4 py-2.5 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:bg-white focus:outline-none transition text-sm">
                        <option value="1">Yes</option>
                        <option value="0" selected>No</option>
                    </select>
                </div>

                <!-- Region -->
                <div>
                    <label class="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1">Region</label>
                    <select name="region" class="w-full px-4 py-2.5 bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:bg-white focus:outline-none transition text-sm">
                        <option value="0">Southwest</option>
                        <option value="1">Southeast</option>
                        <option value="2">Northwest</option>
                        <option value="3">Northeast</option>
                    </select>
                </div>
            </div>

            <!-- Submit Button -->
            <button type="submit" class="w-full mt-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-medium py-3 rounded-lg shadow-md hover:from-blue-700 hover:to-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition duration-150 text-sm tracking-wide cursor-pointer">
                Predict Insurance Cost
            </button>
        </form>

        {% if prediction_text %}
        <!-- Result Box -->
        <div class="bg-blue-50 border-t border-blue-100 p-6 text-center">
            <h2 class="text-xs font-bold text-blue-600 uppercase tracking-wider">Estimated Insurance Charges</h2>
            <p class="text-3xl font-extrabold text-gray-900 mt-2">{{ prediction_text }}</p>
        </div>
        {% endif %}
    </div>

    <!-- Footer -->
    <footer class="text-center text-xs text-gray-500 mt-8">
        Powered by Flask & Render &bull; Medical Insurance Predictor
    </footer>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form inputs matching your model's expected order
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])

        # Prepare feature array for prediction
        features = np.array([[age, sex, bmi, children, smoker, region]])
        
        # Predict using the loaded model
        prediction = model.predict(features)
        output = round(prediction[0], 2)

        return render_template_string(HTML_TEMPLATE, prediction_text=f"${output:,.2f}")
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
