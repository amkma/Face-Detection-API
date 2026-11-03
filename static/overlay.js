// Draws what the detector actually found on top of the preview image.
// Previously the page rendered numbers with no box: a caller could not tell a
// strict threshold from a misplaced detection.
(function () {
  var NAMES = ["eye L", "eye R", "mid", "mouth L", "mouth R"];

  function draw(data) {
    var wrap = document.getElementById("overlay");
    var cv = document.getElementById("boxCanvas");
    var legend = document.getElementById("boxLegend");
    var shown = document.getElementById("preview");
    if (!wrap || !cv || !legend) return;
    wrap.style.display = "none";
    legend.textContent = "";
    var faces = (data && data.faces && data.faces.length) ? data.faces : [];
    if (!faces.length && data && data.facial_area && data.landmarks) {
      faces = [{ facial_area: data.facial_area, landmarks: data.landmarks }];
    }
    if (data && data.face_detected === false) {
      legend.textContent = "No box: the detector found no face in this image.";
      legend.style.display = "block";
      return;
    }
    if (!faces.length || !shown || !shown.naturalWidth) return;

    var w = shown.clientWidth;
    var h = Math.round(shown.naturalHeight * (w / shown.naturalWidth));
    var dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    cv.style.width = w + "px";
    cv.style.height = h + "px";
    cv.style.left = shown.offsetLeft + "px";
    cv.style.top = shown.offsetTop + "px";
    wrap.style.display = "block";

    var k = w / shown.naturalWidth;
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    faces.forEach(function (face) {
      var a = face.facial_area;
      if (!a) return;
      var x = a.x * k;
      var y = a.y * k;
      var bw = a.width * k;
      var bh = a.height * k;
      var ok = !(data.checks || []).length || (data.failed_checks || []).length === 0;

      ctx.lineWidth = Math.max(2, Math.round(w / 220));
      ctx.strokeStyle = ok ? "#1fbf75" : "#e5484d";
      ctx.fillStyle = ok ? "rgba(31,191,117,0.12)" : "rgba(229,72,77,0.12)";
      ctx.fillRect(x, y, bw, bh);
      ctx.strokeRect(x, y, bw, bh);

      var label = Math.round(a.width) + "x" + Math.round(a.height);
      ctx.font = "600 " + Math.max(11, Math.round(w / 42)) + "px ui-sans-serif, system-ui, sans-serif";
      var tw = ctx.measureText(label).width + 10;
      var th = Math.max(16, Math.round(w / 34));
      var ly = y - th < 0 ? y : y - th;
      ctx.fillStyle = ok ? "#1fbf75" : "#e5484d";
      ctx.fillRect(x, ly, tw, th);
      ctx.fillStyle = "#fff";
      ctx.textBaseline = "middle";
      ctx.fillText(label, x + 5, ly + th / 2);

      (face.landmarks || []).forEach(function (pt, index) {
        var px = pt[0] * k;
        var py = pt[1] * k;
        var r = Math.max(2, Math.round(w / 200));
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = "#ffd400";
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = "rgba(0,0,0,0.55)";
        ctx.stroke();
        if ((face.landmarks || []).length === 5) {
          ctx.fillStyle = "#111";
          ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
          ctx.fillText(NAMES[index], px + 4, py - 4);
        }
      });
    });
  }

  window.FaceBoxOverlay = { draw: draw };
})();
