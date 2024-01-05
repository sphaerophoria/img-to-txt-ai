function cloneFullImg() {
  const template = document.getElementById("full-img-template");
  return template.content.cloneNode(true).firstElementChild;
}

class InputImage {
  constructor(parentElement) {
    let div = cloneFullImg();
    this.img = div.querySelector("img");
    parentElement.appendChild(div);
    this.update(0);
  }

  update(id) {
    this.img.src = "/input?id=" + id;
  }
}

class OutputImage {
  constructor(parentElement, initial_metric) {
    let div = cloneFullImg();
    this.img = div.querySelector("img");
    this.id = 0;
    this.metric = initial_metric;
    parentElement.appendChild(div);
    this.update();
  }

  updateId(id) {
    this.id = id;
    this.update();
  }

  updateMetric(metric) {
    this.metric = metric;
    this.update();
  }

  update() {
    this.img.src = "/output?id=" + this.id + "&metric=" + this.metric;
  }
}

class ImageUpdater {
  constructor(inputImage, outputImage, sampler, glyphList) {
    this.id = 0;
    this.inputImage = inputImage;
    this.outputImage = outputImage;
    this.sampler = sampler;
    this.glyphList = glyphList;
  }

  prev() {
    this.id -= 1;
    this.update();
  }

  next() {
    this.id += 1;
    this.update();
  }

  update() {
    this.inputImage.update(this.id);
    this.outputImage.updateId(this.id);
    this.glyphList.updateImage(this.id);
    this.sampler.updateImageId(this.id);
  }
}

class Sampler {
  constructor(target, glyphList, sample_width, sample_height, initialMetric) {
    this.glyphList = glyphList;
    this.picture_x = 0;
    this.picture_y = 0;
    this.target = target;

    this.div = document.createElement("div");
    this.div.classList.add("sample_view");
    this.div.style.position = "absolute";

    this.id = 0;
    this.metric = initialMetric;

    target.addEventListener("load", () => {
      const widthRatio = this.getWidthRatio();
      const heightRatio = this.getHeightRatio();
      this.div.style.width = "" + sample_width * widthRatio + "px";
      this.div.style.height = "" + sample_height * heightRatio + "px";

      this.update(this.id);
    });

    target.parentElement.insertBefore(this.div, target);
    target.onclick = this.onClick.bind(this);
    this.update(this.id);
  }

  onClick(ev) {
    // All APIs are relative to the top left corner of the box, but clicking in the
    // center feels more natural. Find the top left corner relative to the center, and
    // pretend we clicked there
    const pageX = ev.pageX - this.div.offsetWidth / 2.0;
    const pageY = ev.pageY - this.div.offsetHeight / 2.0;
    const widthRatio = this.getWidthRatio();
    const heightRatio = this.getHeightRatio();

    this.picture_x = (pageX - ev.target.offsetLeft) / widthRatio;
    this.picture_y = (pageY - ev.target.offsetTop) / heightRatio;

    this.div.style.left = "" + pageX + "px";
    this.div.style.top = "" + pageY + "px";
    this.update(this.id);
  }

  updateImageId(id) {
    this.id = id;
    this.update();
  }

  updateMetric(metric) {
    this.metric = metric;
    this.update();
  }

  update() {
    document.getElementById("sample-img-input").src =
      "/sample_input?x=" +
      this.picture_x +
      "&y=" +
      this.picture_y +
      "&image_id=" +
      this.id;
    document.getElementById("sample-img-output").src =
      "/sample_output?x=" +
      this.picture_x +
      "&y=" +
      this.picture_y +
      "&metric=" +
      this.metric +
      "&image_id=" +
      this.id;
    this.glyphList.updatePos(this.picture_x, this.picture_y);
  }

  getWidthRatio() {
    return this.target.width / this.target.naturalWidth;
  }

  getHeightRatio() {
    return this.target.height / this.target.naturalHeight;
  }
}

class GlyphList {
  constructor(num_glyphs, initialMetric) {
    this.divs = [];
    this.id = 0;
    this.x = 0;
    this.y = 0;
    this.metric = initialMetric;
    const glyphList = document.getElementById("glyphs");
    let templateCard = document.getElementById("glyph-card-template");
    for (let glyph_num = 0; glyph_num < num_glyphs; glyph_num++) {
      let div = templateCard.content.cloneNode(true).firstElementChild;

      let img = div.querySelector("img");
      img.src = "/glyphs/" + glyph_num;
      div = glyphList.appendChild(div);
      this.divs.push(div);
    }
  }

  async updateMetric(metric) {
    this.metric = metric;
    await this.update();
  }

  async updateImage(id) {
    this.id = id;
    await this.update();
  }

  async updatePos(x, y) {
    this.x = x;
    this.y = y;
    await this.update();
  }

  async update() {
    let scores = await fetch(
      "/sample_metadata?x=" +
        this.x +
        "&y=" +
        this.y +
        "&metric=" +
        this.metric +
        "&image_id=" +
        this.id,
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

    const glyphList = this.divs[0].parentElement;
    glyphList.innerHTML = "";

    for (let i = 0; i < scores.length; ++i) {
      glyphList.appendChild(this.divs[indexes[i]]);
    }
  }
}

function initImageFlipButtons(imageUpdater) {
  document.getElementById("next-img").onclick = () => imageUpdater.next();
  document.getElementById("prev-img").onclick = () => imageUpdater.prev();
}

function initLabelMetrics(labelMetrics, glyphList, sampler, outputImage) {
  const label_metrics_elem = document.getElementById("label-metric-selection");
  for (const label_metric of labelMetrics) {
    const option = document.createElement("option");
    option.value = label_metric;
    option.innerHTML = label_metric;

    label_metrics_elem.appendChild(option);
  }

  label_metrics_elem.onchange = (ev) => {
    outputImage.updateMetric(ev.target.value);
    sampler.updateMetric(ev.target.value);
    glyphList.updateMetric(ev.target.value);
  };
  label_metrics_elem.dispatchEvent(new Event("change"));
}

async function init() {
  let sampleSizePromise = fetch("/sample_size").then((response) =>
    response.json(),
  );
  let numGlyphsPromise = fetch("/glyphs").then((response) => response.json());
  let labelMetricsPromise = fetch("/label_metrics").then((response) =>
    response.json(),
  );

  let [sampleSize, glyphsResponse, labelMetrics] = await Promise.all([
    sampleSizePromise,
    numGlyphsPromise,
    labelMetricsPromise,
  ]);

  let glyphList = new GlyphList(glyphsResponse.num_glyphs, labelMetrics[0]);

  const fullImageContainer = document.getElementById("full-img-comp");
  let inputImage = new InputImage(fullImageContainer);
  let outputImage = new OutputImage(fullImageContainer, labelMetrics[0]);

  const sampler = new Sampler(
    inputImage.img,
    glyphList,
    sampleSize.width,
    sampleSize.height,
    labelMetrics[0],
  );
  const imageUpdater = new ImageUpdater(
    inputImage,
    outputImage,
    sampler,
    glyphList,
  );

  initImageFlipButtons(imageUpdater);
  initLabelMetrics(labelMetrics, glyphList, sampler, outputImage);
}

window.addEventListener("DOMContentLoaded", function () {
  init();
});
