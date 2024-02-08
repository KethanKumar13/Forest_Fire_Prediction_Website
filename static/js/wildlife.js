const submitBtn = document.getElementById('submitBtn');

submitBtn.addEventListener('click', () => {
  const city = document.getElementById('city').value;
  
  // Fetch wildlife data based on the city (simulated data)
  const wildlifeData = fetchDataFromAPI(city); // Assume this function fetches data from an API
  
  // Call a function to draw the pie chart with the fetched wildlife data
  drawPieChart(wildlifeData);
});

function fetchDataFromAPI(city) {
  // Simulated data, replace this with actual API call
  // For example, you can use fetch() to get data from a server
  const wildlifeData = {
    lions: Math.floor(Math.random() * 100),
    tigers: Math.floor(Math.random() * 100),
    bears: Math.floor(Math.random() * 100)
  };

  // For demonstration purposes, we generate random data here
  return wildlifeData;
}

function drawPieChart(data) {
  const canvas = document.getElementById('pieChart');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const values = Object.values(data);
  const colors = ['#ff9999', '#66b3ff', '#99ff99']; // Example colors for the segments

  const total = values.reduce((acc, value) => acc + value, 0);
  let startAngle = 0;

  for (let i = 0; i < values.length; i++) {
    const sliceAngle = (2 * Math.PI * values[i]) / total;
    ctx.beginPath();
    ctx.arc(canvas.width / 2, canvas.height / 2, canvas.width / 3, startAngle, startAngle + sliceAngle);
    ctx.lineTo(canvas.width / 2, canvas.height / 2);
    ctx.fillStyle = colors[i];
    ctx.fill();

    const textX = canvas.width / 2 + (canvas.width / 3) * Math.cos(startAngle + sliceAngle / 2);
    const textY = canvas.height / 2 + (canvas.width / 3) * Math.sin(startAngle + sliceAngle / 2);

    ctx.fillStyle = '#000'; // Text color
    ctx.font = '14px Arial';
    ctx.fillText(`${values[i]}`, textX, textY);

    startAngle += sliceAngle;
  }
}

//Toggle Button
document.getElementById("toggleButton").addEventListener("click", function() {
  document.getElementById("navbar").style.width = "250px";
});

document.addEventListener("click", function(event) {
  if (event.target !== document.getElementById("toggleButton") && event.target.closest(".navbar") === null) {
      document.getElementById("navbar").style.width = "0";
  }
});
