(function () {
  "use strict";

  var COLOR_OK = "#1fbf75";
  var COLOR_BAD = "#e5484d";
  var DOT = "#ffd400";
  var DOT_EDGE = "rgba(0,0,0,0.6)";
  var NAMES = ["eye L", "eye R", "mid", "mouth L", "mouth R"];

  function failedNames(d) {
    if (!d) { return []; }
    if (Array.isArray(d.failed_checks)) { return d.failed_checks; }
    if (d.failed_checks && Array.isArray(d.failed_checks.names)) { return d.failed_checks.names; }
    if (Array.isArray(d.failed)) { return d.failed; }
    return [];
  }

  // A response may describe one face at the top level or a list of faces.
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

  // Accept [[x, y], ...] and [{x, y} | {x, "y": ...}, ...] alike.
  function pointsOf(face) {
    var raw = face.landmarks;
    if (!raw && face.kps) { raw = face.kps; }
    if (!Array.isArray(raw)) { return []; }
    var out = [];
    raw.forEach(function (pt) {
      if (!pt) { return; }
      var x = Array.isArray(pt) ? pt[0] : pt.x;
      var y = Array.isArray(pt) ? pt[1] : pt.y;
      x = Number(x);
      y = Number(y);
      if (!isFinite(x) || !isFinite(y)) { return; }
      out.push([x, y]);
    });
    return out;
  }

  function report(text) {
    var legend = document.getElementById("boxLegend");
    if (!legend) { return; }
    legend.textContent = String(text);
  }

  function paint(data) {
    var cv = document.getElementById("boxCanvas");
    var stage = document.getElementById("boxStage");
    var shown = document.getElementById("preview");
    if (!cv) { return; }
    if (!stage) { return; }
    if (!shown) { return; }

    var faces = facesOf(data);
    stage.style.display = "block";

    // A cleared or rejected frame must wipe the old drawing: leaving the last
    // verdict's box up mislabels the picture that is on screen now.
    function blank(text) {
      cv.width = 0;
      cv.height = 0;
      cv.style.width = "0px";
      cv.style.height = "0px";
      if (text) { report(text); }
    }

    if (data && data.face_detected === false) {
      blank("No box: the detector found no face in this image.");
      return;
    }
    if (!faces.length) {
      blank("");
      return;
    }

    var nat = shown.naturalWidth;
    if (!nat || !shown.clientWidth) { return; }

    // Read where the image actually landed instead of assuming its displayed
    // width equals its natural width; a stretched picture would offset the box.
    var rect = shown.getBoundingClientRect();
    var w = rect.width;
    var h = rect.height;
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
    var dots = 0;

    faces.forEach(function (face) {
      var a = face.facial_area;
      if (a) {
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
      }

      var pts = pointsOf(face);
      dots += pts.length;
      var radius = Math.max(2.5, w / 160);
      pts.forEach(function (pt, index) {
        var px = pt[0] * k;
        var py = pt[1] * k;

        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fillStyle = DOT;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = DOT_EDGE;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(px, py, radius / 3.5, 0, Math.PI * 2);
        ctx.fillStyle = "#111";
        ctx.fill();

        // Small dots on a big face: every point labelled hides the features it
        // is marking, so skip the name when a label would land on another dot.
        var crowded = pts.some(function (other) {
          if (other === pt) { return false; }
          var gap = Math.hypot(other[0] - pt[0], other[1] - pt[1]) * k;
          return gap < radius * 7;
        });
        if (!crowded) {
          ctx.fillStyle = "#111";
          ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
          ctx.textBaseline = "bottom";
          ctx.fillText(NAMES[index] || String(index + 1), px + 5, py - 4);
        }
      });
    });

    var first = faces[0].facial_area || {};
    var size = Math.round(first.width || 0) + "x" + Math.round(first.height || 0);
    var tail = dots ? " - " + dots + " points" : "";
    if (failed.length) {
      report("Box " + size + " px" + tail + ", failed: " + failed.join(", "));
    } else {
      report("Box " + size + " px" + tail + ", all checks passed");
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

  function clear() {
    var cv = document.getElementById("boxCanvas");
    if (cv) { cv.width = 0; }
  }
  window.FaceBoxOverlay = { draw: draw, clear: clear };
})();
