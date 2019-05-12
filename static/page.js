$(document).ready(function () {
    $("#list-name li").click(function () {

        // alert($(this).text()); // gets text contents of clicked li
        alert('kak');
    });

    function show_details(data) {
        var table = "";

        for (i in data) {
            row = "<tr>";
            row += "<td>" + data[i]['content'] + "</td>";
            row += "<td>" + data[i]['quality'] + "</td>";
            row += "<td>" + data[i]['location'] + "</td>";
            row += "<td>" + data[i]['price'] + "</td>";
            row += "<td>" + data[i]['service'] + "</td>";
            row += "<td>" + data[i]['space'] + "</td>";
            row += "</td>";

            table += row;
        }

        $('#content-details-table').html(table);
    }
});