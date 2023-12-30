function getLabelMetric() {
  return document.getElementById("label-metric-selection").value;
}
class Sampler {
  constructor(target, glyph_list, sample_width, sample_height) {
    this.glyph_list = glyph_list;
    this.picture_x = 0;
    this.picture_y = 0;

    this.div = document.createElement("div");
    this.div.classList.add("sample_view");
    this.div.style.position = "absolute";

    // FIXME: add listener for img size change
    this.width_ratio = target.width / target.naturalWidth;
    this.height_ratio = target.height / target.naturalHeight;

    this.div.style.width = "" + sample_width * this.width_ratio + "px";
    this.div.style.height = "" + sample_height * this.height_ratio + "px";

    target.parentElement.insertBefore(this.div, target);
    target.onclick = this.onClick.bind(this);
    this.update(0, 0);
  }

  onClick(ev) {
    // All APIs are relative to the top left corner of the box, but clicking in the
    // center feels more natural. Find the top left corner relative to the center, and
    // pretend we clicked there
    const pageX = ev.pageX - this.div.offsetWidth / 2.0;
    const pageY = ev.pageY - this.div.offsetHeight / 2.0;

    this.picture_x = (pageX - ev.target.offsetLeft) / this.width_ratio;
    this.picture_y = (pageY - ev.target.offsetTop) / this.height_ratio;

    this.div.style.left = "" + pageX + "px";
    this.div.style.top = "" + pageY + "px";
    this.update();
  }

  update() {
    document.getElementById("sample-img-input").src =
      "/sample_input?x=" + this.picture_x + "&y=" + this.picture_y;
    document.getElementById("sample-img-output").src =
      "/sample_output?x=" +
      this.picture_x +
      "&y=" +
      this.picture_y +
      "&metric=" +
      getLabelMetric();
    this.glyph_list.update(this.picture_x, this.picture_y);
  }
}

class GlyphList {
  constructor(num_glyphs) {
    this.divs = [];
    const glyph_list = document.getElementById("glyphs");
    let templateCard = document.getElementById("glyph-card-template");
    for (let glyph_num = 0; glyph_num < num_glyphs; glyph_num++) {
      let div = templateCard.content.cloneNode(true).firstElementChild;

      let img = div.querySelector("img");
      img.src = "/glyphs/" + glyph_num;
      div = glyph_list.appendChild(div);
      this.divs.push(div);
    }
  }

  async update(x, y) {
    let scores = await fetch(
      "/sample_metadata?x=" + x + "&y=" + y + "&metric=" + getLabelMetric(),
    );
    scores = await scores.json();

    for (let i = 0; i < scores.length; ++i) {
      this.divs[i].querySelector(".glyph-score").innerHTML =
        scores[i].toPrecision(4);
    }

    let indexes = Array.from({ length: scores.length }, (e, i) => i);
    indexes.sort(function (a, b) {
      if (scores[a] < scores[b]) {
        return -1;
      }
      if (scores[a] > scores[b]) {
        return 1;
      }

      return 0;
    });

    const glyph_list = this.divs[0].parentElement;
    glyph_list.innerHTML = "";

    for (let i = 0; i < scores.length; ++i) {
      glyph_list.appendChild(this.divs[indexes[i]]);
    }
  }
}

async function onLabelMetricSelection(ev, sampler) {
  const outputImg = document.getElementById("output-img");
  outputImg.src = "/output?metric=" + ev.target.value;
  sampler.update();
}

function initLabelMetrics(label_metrics, sampler) {
  const label_metrics_elem = document.getElementById("label-metric-selection");
  for (const label_metric of label_metrics) {
    const option = document.createElement("option");
    option.value = label_metric;
    option.innerHTML = label_metric;

    label_metrics_elem.appendChild(option);
  }

  label_metrics_elem.onchange = (ev) => {
    onLabelMetricSelection(ev, sampler);
  };
  label_metrics_elem.dispatchEvent(new Event("change"));
}

async function init() {
  const img = document.getElementById("input-img");

  let sample_size_promise = fetch("/sample_size").then((response) =>
    response.json(),
  );
  let num_glyphs_promise = fetch("/glyphs").then((response) => response.json());
  let label_metrics_promise = fetch("/label_metrics").then((response) =>
    response.json(),
  );

  let [sample_size, glyphs_response, label_metrics] = await Promise.all([
    sample_size_promise,
    num_glyphs_promise,
    label_metrics_promise,
  ]);

  let glyph_list = new GlyphList(glyphs_response.num_glyphs);

  const sampler = new Sampler(
    img,
    glyph_list,
    sample_size.width,
    sample_size.height,
  );
  initLabelMetrics(label_metrics, sampler);
}

window.addEventListener("DOMContentLoaded", function () {
  init();
});
