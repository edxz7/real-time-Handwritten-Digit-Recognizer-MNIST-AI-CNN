/*
conde modified from:

 License: Apache-2.0
 Author: Eduardo Ch. Colorado
 E-mail: edxz7c@gmail.com
*/

let el = x => document.getElementById(x);

// Predict button callback
$("#predict").click(function(){  
  // Change status indicator
  $("#status").removeClass().toggleClass("fa fa-spinner fa-spin");
  let canvas = document.getElementById('canvas');
  let canvas_data = canvas.toDataURL('png');
  // Get canvas contents as url
  // var fac = (1.) / 13.; 
  // var canvas_data = canvas.toDataURLWithMultiplier('png', fac);
  $.ajax({
    type: 'POST',
    url: '/analyze',
    data: canvas_data,
    contentType: false,
    cache: false,
    processData: false,
    async: true,
    success: function (json) {
      // data = JSON.parse(data)
      console.log(typeof json);
      if (json.result) {
        $("#status").removeClass().toggleClass("fa fa-check");
        $('#svg-chart').show();
        updateChart(json.data);
      } else {
          $("#status").removeClass().toggleClass("fa fa-exclamation-triangle");
          console.log('Script Error: ' + json.error)
      }
    },
    error: function (xhr, textStatus, error) {
          $("#status").removeClass().toggleClass("fa fa-exclamation-triangle");
          console.log("POST Error: " + xhr.responseText + ", " + textStatus + ", " + error);
        }
      });
    });

// Iniitialize d3 bar chart
$('#svg-chart').hide();
var labels = ['0','1','2','3','4','5','6','7','8','9'];
var zeros = [0,0,0,0,0,0,0,0,0,0,0];

// chartOptions = [{
//   "captions": [{"1":"1","2":"2","3":"3","4":"4","5":"5","6":"6","7":"7","8":"8","9":"9"}]
// }]

var margin = {top: 0, right: 0, bottom: 20, left: 0},
    width = 360 - margin.left - margin.right,
    height = 180 - margin.top - margin.bottom;

var svg = d3.select("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", 
          "translate(" + margin.left + "," + margin.top + ")");

var x = d3.scale.ordinal()
    .rangeRoundBands([0, width], .1)
    .domain(labels);
    
var y = d3.scale.linear()
          .range([height, 0])
          .domain([0,1]);  

var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .tickSize(0);

svg.selectAll(".bar")
    .data(zeros)
    .enter().append("rect")
    .attr("class", "bar")
    .attr("x", function(d, i) { return x(i); })
    .attr("width", x.rangeBand())
    .attr("y", function(d) { return y(d); })
    .attr("height", function(d) { return height - y(d); })
    .style('background', 'black');

svg.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0," + height + ")")
    .call(xAxis);

// Update chart data
function updateChart(data) {
  d3.selectAll("rect")
    .data(data)
    .transition()
    .duration(500)
    .attr("y", function(d) { return y(d); })
    .attr("height", function(d) { return height - y(d); })
    .style('background', 'black');
}
    
///////////////////

//   let xAxisScale = d3.scaleBand()
//     .domain(['3', '7'])
//     .rangeRound([0,( width - 40)])
//   let yAxisScale = d3.scaleLinear()
//     .rangeRound([0, (40 - height)])

//   let xAxis = d3.axisBottom()
//     .scale(xAxisScale)
//   let yAxis = d3.axisLeft()
//     .scale(yAxisScale)

//   let animateDuration = 200
//   let animateDelay = 100

//   // Prevents duplication by removing svg on each re render
//   let removeChart = d3.selectAll('svg').remove();

//   let mainChart = d3.select('.mainChart')
//   mainChart.append('svg')
//       .attr('width', width)
//       .attr('height', height)
//       .style('background', 'black')
//       .selectAll('rect')
//         .data(this.props.outputs)
//           .enter().append('rect')
//           .style('fill', 'white')
//           .attr('width', barWidth)
//           .attr('height', 0)
//           .attr('x', function(d, i) {
//             return ((i + 1) * 180) - 70
//           })
//           .attr('y', function(d) {
//             console.log(height * d)
//             return (height - ((height - 40) * d) - 20)
//           })

//   d3.select('svg').append('g')
//     .attr("transform", "translate(40, " + (height - 20) + ")")
//     .style('stroke', 'white')
//     .call(xAxis)
//   d3.select('svg').append('g')
//     .attr('transform', 'translate(40, ' + (height - 20) + ')')
//     .style('stroke', 'white')
//     .call(yAxis)

//   let allRects = d3.selectAll('rect')
//     allRects.transition()
//       .attr('height', function(d) {
//         return (height - 40) * d
//       })
//       .ease(d3.easeLinear)
//       .duration(animateDuration)
// }
  // render() {
  //   return (
  //     <div className='mainChart' style={{marginLeft: '20px'}}>
  //       {this.mainChartRun()}
  //     </div>
  //   )
  // }
// }
