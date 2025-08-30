document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('health-form');

    form.addEventListener('submit', (e) => {
        e.preventDefault();

        // Show loading spinner
        const resultSection = document.getElementById('result-section');
        const riskLevel = document.getElementById('risk-level');
        const resultText = document.getElementById('result-text');
        riskLevel.textContent = '';
        resultText.textContent = 'Loading...';
        resultSection.style.display = 'block';

        // Basic validation to ensure fields are not empty
        const age = document.getElementById('age').value;
        const bp = document.getElementById('bp').value;
        const weight = document.getElementById('weight').value;
        const hr = document.getElementById('hr').value;
        const history = document.getElementById('history').value;

        if (!age || !bp || !weight || !hr || !history) {
            alert('Please fill out all fields.');
            return;
        }

        // Map Smoking and PastDiagnosis to expected numeric values
        let smokingVal = document.querySelector('input[name="smoking"]:checked').value;
        smokingVal = smokingVal === 'yes' ? 1 : 0;
        let historyVal = 0;
        if (history === 'none') historyVal = 0;
        else if (history === 'heart-disease') historyVal = 1;
        else if (history === 'hypertension') historyVal = 2;
        else if (history === 'diabetes') historyVal = 3;

        const formData = {
            Age: parseInt(age),
            BP: parseInt(bp),
            Weight: parseInt(weight),
            HeartRate: parseInt(hr),
            Smoking: smokingVal,
            PastDiagnosis: historyVal
        };

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                riskLevel.textContent = 'Error';
                resultText.textContent = data.error;
            } else {
                // Show prediction and confidence
                riskLevel.textContent = data.prediction;
                riskLevel.style.color = data.prediction === 'Healthy' ? '#00b4d8' : '#dc3545';
                resultText.textContent = `Confidence: ${data.confidence}`;
            }
        })
        .catch(error => {
            riskLevel.textContent = 'Error';
            resultText.textContent = 'An error occurred: ' + error.message;
        });
    });

    function displayResult(prediction, confidence) {
        const resultSection = document.getElementById('result-section');
        const riskLevel = document.getElementById('risk-level');
        const resultText = document.getElementById('result-text');

        riskLevel.textContent = prediction === 'Low Risk' ? 'Low Risk' : 'High Risk';
        riskLevel.style.color = prediction === 'Low Risk' ? '#00b4d8' : '#dc3545';
        resultText.textContent = 'This is a preliminary assessment. Please consult with a healthcare professional for detailed evaluation.';
        
        resultSection.style.display = 'block';
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    // Show assessment form and hide hero section on button click
    const startButton = document.querySelector('.btn-primary[href="#assessment-form"]');
    if (startButton) {
        startButton.addEventListener('click', function(e) {
            e.preventDefault();
            document.getElementById('assessment-form').scrollIntoView({ behavior: 'smooth' });
        });
    }
});
