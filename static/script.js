async function predict() {
    let fileInput = document.getElementById("imageInput");
    let file = fileInput.files[0];

    let formData = new FormData();
    formData.append("file", file);

    let response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    let data = await response.json();

    document.getElementById("result").innerText =
        `${data.prediction} (${data.confidence.toFixed(2)})\n${data.explanation}`;
}