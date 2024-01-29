function previewImage() {
    var fileInput = document.getElementById('file');
    var uploadedImg = document.getElementById('uploaded-img');

    var file = fileInput.files[0];
    if (file) {
        var reader = new FileReader();
        reader.onload = function (e) {
            uploadedImg.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
}

function uploadPhoto() {
    var fileInput = document.getElementById('file');
    var uploadedImg = document.getElementById('uploaded-img');

    var file = fileInput.files[0];
    if (file) {
        var formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData,
        })
            .then(response => response.json())
            .then(data => {
                alert('Image uploaded successfully');
                console.log(data);
                // Add any further processing or display logic here
            })
            .catch(error => {
                console.error('Error:', error);
            });
    } else {
        alert('Please choose a file before uploading.');
    }
}


function processImage() {
    var resultSection = document.getElementById('result-section');

    // Placeholder JSON-like result
    var processingResults = {
        detected_items: [
            { name: "peach", total: 5, rotten: 2, percentage: 65 },
            { name: "tomato", total: 4, rotten: 1, percentage: 20 },
            { name: "orange", total: 3, rotten: 1, percentage: 30 }
        ]
    };

    // Create a styled HTML representation of the results as a centered table
    var htmlResults = '<h2>Processing Results:</h2><div class="result-container">';
    htmlResults += '<table style="margin: 0 auto; text-align: center;">';
    htmlResults += '<thead><tr><th>Name</th><th>Total</th><th>Rotten</th><th>Percentage</th></tr></thead>';
    htmlResults += '<tbody>';

    processingResults.detected_items.forEach(item => {
        // Capitalize the first letter of the item's name
        var capitalizedItemName = item.name.charAt(0).toUpperCase() + item.name.slice(1);

        htmlResults += '<tr>';
        htmlResults += '<td>' + capitalizedItemName + '</td>';
        htmlResults += '<td>' + item.total + '</td>';
        htmlResults += '<td>' + item.rotten + '</td>';
        htmlResults += '<td>' + item.percentage + '%</td>';
        htmlResults += '</tr>';
    });

    htmlResults += '</tbody></table></div>';

    resultSection.innerHTML = htmlResults;
}