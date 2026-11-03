// Draws what the detector found on top of the preview image.
// The canvas sits in normal flow inside #boxStage, so it tracks the picture
// when the window is resized instead of being placed by offset arithmetic.
(function () {
  var NAMES = ["eye L", "eye R", "mid", "mouth L", "mouth R"];
  var COLOR_OK = "#1fbf75";
  var COLOR_BAD = "#e5484d";

  function report(text) {
    var legend = document.getElementById("boxLegend");
    if (legend) {
      legend.textContent = text;
    }
  }

  function clear() {
    var cv = document.getElementById("boxCanvas");
    if (!cv) { return; }
    cv.width = 0;
    cv.height = 0;
    cv.style.width = "0px";
    cv.style.height = "0px";
    report("");
  }

  function failedNames(data) {
    if (data && data.failed_checks) {
      return data.failed_checks;
    }
    return [];
  }

  function facesOf(d) {
    var list = [];
    if (!d) { return list; }
    if (d.facesColumn) { return d.facesColumn; }
    if (d.results && d.results.facesColumn) { return d.results.facesColumn; }
    if (d.faces) { return d.faces; }
    if (d.results && d.results.faces) { return d.results.faces; }
    if (d.facial_area) { list.push(d); }
    return list;
  }

  function paint(data) {
    var cv = document.getElementById("boxCanvas");
    var stage = document.getElementById("boxStage");
    var shown = document.getElementById("preview");
    if (!cv) { return; }
    if (!stage) { return; }
    if (!shown) { return; }
    if (!shown.clientWidth) { return; }

    var faces = facesOf(data);
    if (data && data.face_detected === false) {
      cv.width = 0;
      cv.height = 0;
      cv.style.width = "0px";
      cv.style.height = "0px";
      report("No box: the detector found no face in this image.");
      return;
    }
    if (!faces.length) { return; }

    var nat = shown.naturalWidth;
    if (!nat) { return; }

    // Read where the image actually landed instead of assuming its displayed
    // width equals its natural width; a stretched picture would offset the box.
    var box = shown.getBoundingClientRect();
    var w = box.width;
    var h = box.height;
    var k = w / nat;

    var dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    cv.style.width = w + "px";
    cv.style.height = h + "px";

    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    var failed = failedNames(data);
    var ok = failed.length === 0;
    var stroke = ok ? COLOR_OK : COLOR_BAD;
    var fill = ok ? "rgba(31,191,117,0.12)" : "rgba(229,72,77,0.12)";
    var font = Math.max(11, Math.round(w / 42));

    faces.forEach(function (face) {
      var a = face.facial_area;
      if (!a) { return; }
      var x = a.x * k;
      var y = a.y * k;
      var bw = a.width * k;
      var bh = a.height * k;

      ctx.lineWidth = Math.max(2, Math.round(w / 220));
      ctx.strokeStyle = stroke;
      ctx.fillStyle = fill;
      ctx.fillRect(x, y, bw, bh);
      ctx.strokeRect(x, y, bw, bh);

      var label = Math.round(a.width) + "x" + Math.round(a.height);
      ctx.font = "600 " + font + "px ui-sans-serif, system-ui, sans-serif";
      var tw = ctx.measureText(label).width + 10;
      var th = Math.max(16, Math.round(w / 34));
      var ly = (y - th > 0) ? (y - th) : y;
      ctx.fillStyle = stroke;
      ctx.fillRect(x, ly, tw, th);
      ctx.fillStyle = "#fff";
      ctx.textBaseline = "middle";
      ctx.fillText(label, x + 5, ly + th / 2);
    });

    var first = faces[0].facial_area;
    var size = Math.round(first.width) + "x" + Math.round(first.height);
    if (failed.length) {
      report("Box " + size + " px, failed: " + failed.join(", "));
    } else {
      report("Box " + size + " px, all checks passed");
    }
  }

  function draw(data) {
    var shown = document.getElementById("preview");
    var stage = document.getElementById("boxStage");
    var legend = document.getElementById("boxLegend");
    if (!shown) { return; }
    if (!stage) { return; }
    stage.style.display = "block";
    if (legend) { legend.style.display = "block"; }
    if (shown.naturalWidth) { paint(data); return; }
    var done = false;
    function once() {
      if (done) { return; }
      done = true;
      shown.removeEventListener("load", once);
      shown.removeEventListener("error", once);
      if (shown.naturalWidth) { paint(data); } else { report("Image could not be loaded."); }
    }
    shown.addEventListener("load", once);
    shown.addEventListener("error", once);
  }

  window.FaceBoxOverlay = { draw: draw, clear: clear };
})();
