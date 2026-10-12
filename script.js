
const form = document.getElementById("news-form");
const button = document.getElementById("analyze-button");

form.addEventListener("submit", function () {
    if (!form.checkValidity()) {
        return;
    }

    button.textContent = "Analyzing News...";
    button.disabled = true;
});
