(() => {
  "use strict";
  const API_BASE = "http://127.0.0.1:8000";
  const categorical = {
    protocol_type: ["tcp", "udp", "icmp"],
    service: ["http", "private", "domain_u", "smtp", "ftp_data", "other", "eco_i", "ecr_i", "telnet", "ftp", "finger", "pop_3", "auth", "imap", "ssh", "ntp_u", "uucp", "remote_job", "time", "domain", "name", "whois", "gopher", "rje", "link", "login", "printer", "shell", "sql_net", "bgp", "courier", "csnet_ns", "ctf", "daytime", "discard", "echo", "exec", "hostnames", "http_443", "kshell", "netbios_dgm", "netbios_ns", "netbios_ssn", "netstat", "nnsp", "nntp", "pm_dump", "red_i", "sunrpc", "supdup", "systat", "uucp_path", "vmnet", "Z39_50"],
    flag: ["SF", "S0", "REJ", "RSTR", "RSTO", "S1", "S2", "S3", "SH", "OTH"]
  };
  const groups = [
    {title:"Connection basics", subtitle:"01", fields:["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent"]},
    {title:"Content and authentication", subtitle:"02", fields:["hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","is_host_login","is_guest_login"]},
    {title:"Connection statistics", subtitle:"03", fields:["count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate"]},
    {title:"Destination host statistics", subtitle:"04", fields:["dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]}
  ];
  const excluded = new Set(["num_outbound_cmds"]);
  const allFields = groups.flatMap(group => group.fields).filter(name => !excluded.has(name));
  const formContainer = document.getElementById("feature-groups");
  const form = document.getElementById("connection-form");
  const submitButton = document.getElementById("analyze-button");
  const modeButtons = [...document.querySelectorAll(".mode-button")];
  let mode = "binary";
  let apiOnline = false;
  let modelsReady = false;

  function prettyName(name) {
    return name.replaceAll("_", " ").replace(/\b\w/g, char => char.toUpperCase());
  }

  function renderForm() {
    formContainer.innerHTML = "";
    groups.forEach(group => {
      const section = document.createElement("section");
      section.className = "feature-group";
      const title = document.createElement("h3");
      title.className = "group-title";
      title.innerHTML = '<span class="group-number">' + group.subtitle + '</span>' + group.title;
      const fields = document.createElement("div");
      fields.className = "group-fields";
      group.fields.filter(name => !excluded.has(name)).forEach(name => {
        const wrap = document.createElement("div");
        wrap.className = "field";
        const id = "feature-" + name;
        const label = document.createElement("label");
        label.htmlFor = id;
        const labelText = document.createElement("span");
        labelText.textContent = prettyName(name);
        const code = document.createElement("code");
        code.textContent = name;
        label.append(labelText, code);
        let input;
        if (categorical[name]) {
          input = document.createElement("select");
          categorical[name].forEach(value => {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = value;
            input.append(option);
          });
        } else {
          input = document.createElement("input");
          input.type = "number";
          input.step = ["serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"].includes(name) ? "0.01" : "1";
          input.min = "0";
          input.value = "0";
        }
        input.id = id;
        input.name = name;
        input.required = true;
        input.setAttribute("aria-label", prettyName(name));
        wrap.append(label, input);
        fields.append(wrap);
      });
      section.append(title, fields);
      formContainer.append(section);
    });
    document.querySelector(".feature-count").innerHTML = allFields.length + ' <span>FEATURES</span>';
  }

  function collectRecord() {
    const record = {};
    for (const name of allFields) {
      const input = form.elements.namedItem(name);
      if (!input) throw new Error("Missing form control for feature: " + name);
      if (categorical[name]) {
        record[name] = input.value;
      } else {
        const value = input.value.trim();
        if (value === "" || !Number.isFinite(Number(value))) {
          input.classList.add("invalid");
          input.focus();
          throw new Error("Enter a valid number for " + prettyName(name) + ".");
        }
        input.classList.remove("invalid");
        record[name] = Number(value);
      }
    }
    return record;
  }

  function setApiStatus(online, ready, modelsPresent) {
    apiOnline = online;
    modelsReady = Boolean(ready);
    ["api-dot", "sidebar-dot"].forEach(id => {
      const dot = document.getElementById(id);
      dot.classList.toggle("online", online && modelsReady);
      dot.classList.toggle("offline", !online || (online && !modelsReady));
    });
    document.getElementById("sidebar-api-text").textContent = !online ? "Backend unavailable" : modelsReady ? "Connected · models ready" : "Connected · models missing";
    document.getElementById("api-status-title").textContent = !online ? "API unavailable" : modelsReady ? "Backend connected · models ready" : "Backend connected · models not found";
    document.getElementById("api-status-detail").textContent = !online
      ? "Start Uvicorn in your project terminal to enable predictions."
      : modelsReady
        ? "Both saved model artifacts are present and ready for inference."
        : "The API is running, but generate model artifacts before predicting.";
    if (online && modelsPresent) {
      const missing = Object.entries(modelsPresent).filter(([,present]) => !present).map(([task]) => task);
      document.getElementById("api-status-detail").textContent = modelsReady
        ? "Both saved model artifacts are present and ready for inference."
        : "Missing model(s): " + missing.join(", ") + ". Run python -m src.training.save_models.";
    }
    submitButton.disabled = !(online && modelsReady);
    runBatchButton.disabled = !(online && modelsReady && batchValidated);
  }

  async function refreshHealth() {
    const title = document.getElementById("api-status-title");
    title.textContent = "Checking API…";
    try {
      const response = await fetch(API_BASE + "/health", {method:"GET"});
      if (!response.ok) throw new Error("Health check returned HTTP " + response.status);
      const data = await response.json();
      setApiStatus(true, Boolean(data.models_ready), data.models_present || {});
    } catch (error) {
      setApiStatus(false, false, null);
    }
  }

  function showError(message) {
    document.getElementById("result-content").classList.add("hidden");
    document.getElementById("empty-result").classList.add("hidden");
    const box = document.getElementById("result-error");
    box.textContent = message;
    box.classList.remove("hidden");
  }

  function showResult(result) {
    const prediction = result.prediction;
    const isAttack = prediction === "Attack" || ["DoS","Probe","R2L","U2R"].includes(prediction);
    document.getElementById("empty-result").classList.add("hidden");
    document.getElementById("result-error").classList.add("hidden");
    document.getElementById("result-content").classList.remove("hidden");
    document.getElementById("result-label").textContent = isAttack ? "THREAT CLASSIFICATION" : "CONNECTION CLASSIFICATION";
    document.getElementById("result-label").style.color = isAttack ? "var(--red)" : "var(--mint)";
    document.getElementById("result-title").textContent = prediction;
    document.getElementById("result-title").style.color = isAttack ? "var(--red)" : "var(--mint)";
    document.getElementById("result-badge").textContent = mode === "binary" ? "BINARY" : "FIVE CLASS";
    document.getElementById("result-description").textContent = isAttack
      ? "The model classified this record as " + prediction + ". Review the connection in context; a model output alone is not proof of malicious activity."
      : "The model classified this record as normal. This is a prediction, not a guarantee that the connection is safe.";
    const confidence = Number.isFinite(result.confidence) ? result.confidence : null;
    document.getElementById("confidence-value").textContent = confidence === null ? "N/A" : (confidence * 100).toFixed(1) + "%";
    document.getElementById("confidence-bar").style.width = confidence === null ? "0%" : Math.max(0, Math.min(100, confidence * 100)) + "%";
    const attackBlock = document.getElementById("attack-probability-block");
    if (typeof result.attack_probability === "number") {
      attackBlock.classList.remove("hidden");
      document.getElementById("attack-probability").textContent = (result.attack_probability * 100).toFixed(1) + "%";
      document.getElementById("attack-probability-bar").style.width = Math.max(0, Math.min(100, result.attack_probability * 100)) + "%";
    } else {
      attackBlock.classList.add("hidden");
    }
    document.getElementById("result-time").textContent = "Response received";
  }

  modeButtons.forEach(button => button.addEventListener("click", () => {
    mode = button.dataset.mode;
    modeButtons.forEach(item => item.classList.toggle("selected", item === button));
    document.getElementById("result-badge").textContent = mode === "binary" ? "BINARY" : "FIVE CLASS";
  }));

  form.addEventListener("submit", async event => {
    event.preventDefault();
    let record;
    try {
      record = collectRecord();
    } catch (error) {
      showError(error.message);
      return;
    }
    const originalText = submitButton.querySelector("span").textContent;
    submitButton.disabled = true;
    submitButton.querySelector("span").textContent = "Analyzing…";
    document.getElementById("result-content").classList.add("hidden");
    document.getElementById("result-error").classList.add("hidden");
    document.getElementById("empty-result").classList.remove("hidden");
    document.querySelector("#empty-result h3").textContent = "Analyzing connection";
    document.querySelector("#empty-result p").textContent = "Sending the feature record to your local SentriNet API.";
    try {
      const endpoint = mode === "binary" ? "/predict/binary" : "/predict/multiclass";
      const response = await fetch(API_BASE + endpoint, {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({records:[record]})
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = data.detail;
        const message = typeof detail === "string" ? detail : detail?.message || JSON.stringify(detail) || "Request failed.";
        throw new Error("API error (" + response.status + "): " + message);
      }
      if (!Array.isArray(data.predictions) || !data.predictions[0]) throw new Error("The API returned no prediction.");
      showResult(data.predictions[0]);
    } catch (error) {
      showError(error.message.includes("Failed to fetch")
        ? "Could not reach the API. Make sure Uvicorn is running at " + API_BASE + " and refresh the status."
        : error.message);
      document.querySelector("#empty-result h3").textContent = "Awaiting connection";
      document.querySelector("#empty-result p").textContent = "Submit a record to see the model's prediction and confidence score.";
    } finally {
      submitButton.disabled = !(apiOnline && modelsReady);
      submitButton.querySelector("span").textContent = originalText;
    }
  });

  const scenarioPresets = {
    normal: {
      duration:0, protocol_type:"tcp", service:"http", flag:"SF", src_bytes:181, dst_bytes:5450,
      land:0, wrong_fragment:0, urgent:0, hot:0, num_failed_logins:0, logged_in:1,
      num_compromised:0, root_shell:0, su_attempted:0, num_root:0, num_file_creations:0,
      num_shells:0, num_access_files:0, is_host_login:0, is_guest_login:0, count:8, srv_count:8,
      serror_rate:0, srv_serror_rate:0, rerror_rate:0, srv_rerror_rate:0, same_srv_rate:1,
      diff_srv_rate:0, srv_diff_host_rate:0, dst_host_count:9, dst_host_srv_count:9,
      dst_host_same_srv_rate:1, dst_host_diff_srv_rate:0, dst_host_same_src_port_rate:0.11,
      dst_host_srv_diff_host_rate:0, dst_host_serror_rate:0, dst_host_srv_serror_rate:0,
      dst_host_rerror_rate:0, dst_host_srv_rerror_rate:0
    },
    dos: {
      duration:0, protocol_type:"tcp", service:"http", flag:"S0", src_bytes:0, dst_bytes:0,
      land:0, wrong_fragment:0, urgent:0, hot:0, num_failed_logins:0, logged_in:0,
      num_compromised:0, root_shell:0, su_attempted:0, num_root:0, num_file_creations:0,
      num_shells:0, num_access_files:0, is_host_login:0, is_guest_login:0, count:511, srv_count:511,
      serror_rate:0.99, srv_serror_rate:0.99, rerror_rate:0, srv_rerror_rate:0,
      same_srv_rate:1, diff_srv_rate:0, srv_diff_host_rate:0, dst_host_count:255,
      dst_host_srv_count:255, dst_host_same_srv_rate:1, dst_host_diff_srv_rate:0,
      dst_host_same_src_port_rate:0, dst_host_srv_diff_host_rate:0, dst_host_serror_rate:0.99,
      dst_host_srv_serror_rate:0.99, dst_host_rerror_rate:0, dst_host_srv_rerror_rate:0
    },
    probe: {
      duration:0, protocol_type:"tcp", service:"private", flag:"S0", src_bytes:0, dst_bytes:0,
      land:0, wrong_fragment:0, urgent:0, hot:0, num_failed_logins:0, logged_in:0,
      num_compromised:0, root_shell:0, su_attempted:0, num_root:0, num_file_creations:0,
      num_shells:0, num_access_files:0, is_host_login:0, is_guest_login:0, count:30, srv_count:1,
      serror_rate:0.8, srv_serror_rate:0.8, rerror_rate:0.1, srv_rerror_rate:0.1,
      same_srv_rate:0.03, diff_srv_rate:0.97, srv_diff_host_rate:0.95, dst_host_count:255,
      dst_host_srv_count:10, dst_host_same_srv_rate:0.04, dst_host_diff_srv_rate:0.96,
      dst_host_same_src_port_rate:0.01, dst_host_srv_diff_host_rate:0.9, dst_host_serror_rate:0.75,
      dst_host_srv_serror_rate:0.7, dst_host_rerror_rate:0.1, dst_host_srv_rerror_rate:0.1
    },
    login: {
      duration:0, protocol_type:"tcp", service:"ftp", flag:"REJ", src_bytes:350, dst_bytes:1200,
      land:0, wrong_fragment:0, urgent:0, hot:1, num_failed_logins:5, logged_in:0,
      num_compromised:0, root_shell:0, su_attempted:0, num_root:0, num_file_creations:0,
      num_shells:0, num_access_files:0, is_host_login:0, is_guest_login:1, count:10, srv_count:2,
      serror_rate:0, srv_serror_rate:0, rerror_rate:0.8, srv_rerror_rate:0.8,
      same_srv_rate:0.2, diff_srv_rate:0.8, srv_diff_host_rate:0.5, dst_host_count:40,
      dst_host_srv_count:8, dst_host_same_srv_rate:0.2, dst_host_diff_srv_rate:0.8,
      dst_host_same_src_port_rate:0.1, dst_host_srv_diff_host_rate:0.5, dst_host_serror_rate:0,
      dst_host_srv_serror_rate:0, dst_host_rerror_rate:0.7, dst_host_srv_rerror_rate:0.7
    }
  };

  document.getElementById("load-preset").addEventListener("click", () => {
    const presetName = document.getElementById("scenario-preset").value;
    const preset = scenarioPresets[presetName];
    if (!preset || Object.keys(preset).length !== allFields.length) {
      showError("The selected demo preset does not match the 40-feature model schema.");
      return;
    }
    allFields.forEach(name => {
      const input = form.elements.namedItem(name);
      if (input) {
        input.value = String(preset[name]);
        input.classList.remove("invalid");
      }
    });
    document.getElementById("result-content").classList.add("hidden");
    document.getElementById("result-error").classList.add("hidden");
    document.getElementById("empty-result").classList.remove("hidden");
    document.querySelector("#empty-result h3").textContent = "Scenario loaded";
    document.querySelector("#empty-result p").textContent =
      "Illustrative " + document.getElementById("scenario-preset").selectedOptions[0].textContent +
      " values loaded. The model determines the prediction; the scenario name does not force a label.";
  });

  document.getElementById("reset-form").addEventListener("click", () => {
    form.reset();
    form.querySelectorAll('input[type="number"]').forEach(input => {input.value="0";input.classList.remove("invalid");});
    document.getElementById("result-content").classList.add("hidden");
    document.getElementById("result-error").classList.add("hidden");
    document.getElementById("empty-result").classList.remove("hidden");
    document.querySelector("#empty-result h3").textContent = "Awaiting connection";
    document.querySelector("#empty-result p").textContent = "Submit a record to see the model's prediction and confidence score.";
  });
  document.getElementById("refresh-health").addEventListener("click", refreshHealth);


  // Batch CSV workflow: the API validates the full file before any inference.
  const batchFileInput = document.getElementById("batch-file");
  const previewBatchButton = document.getElementById("preview-batch");
  const runBatchButton = document.getElementById("run-batch");
  const batchMessage = document.getElementById("batch-message");
  let selectedBatchFile = null;
  let batchValidated = false;
  let batchPredictions = [];

  function setBatchMessage(message, kind) {
    batchMessage.textContent = message;
    batchMessage.className = "batch-message" + (kind ? " " + kind : "");
  }

  function renderTable(table, headers, rows) {
    table.replaceChildren();
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    headers.forEach(value => {
      const th = document.createElement("th");
      th.textContent = value;
      headRow.append(th);
    });
    thead.append(headRow);
    const tbody = document.createElement("tbody");
    rows.forEach(row => {
      const tr = document.createElement("tr");
      row.forEach(value => {
        const td = document.createElement("td");
        td.textContent = value == null ? "" : String(value);
        tr.append(td);
      });
      tbody.append(tr);
    });
    table.append(thead, tbody);
  }

  function resetBatchResults() {
    batchPredictions = [];
    document.getElementById("batch-results").classList.add("hidden");
    document.getElementById("batch-preview-wrap").classList.add("hidden");
    runBatchButton.disabled = true;
  }

  batchFileInput.addEventListener("change", () => {
    selectedBatchFile = batchFileInput.files[0] || null;
    batchValidated = false;
    resetBatchResults();
    previewBatchButton.disabled = !selectedBatchFile;
    document.getElementById("batch-file-name").textContent =
      selectedBatchFile ? selectedBatchFile.name : "Choose a CSV file";
    setBatchMessage(
      selectedBatchFile ? "File selected. Validate it to inspect the schema and first records." : "Select a CSV to begin.",
      ""
    );
  });

  previewBatchButton.addEventListener("click", async () => {
    if (!selectedBatchFile) return;
    if (selectedBatchFile.size > 5 * 1024 * 1024) {
      setBatchMessage("The CSV exceeds the 5 MB upload limit.", "error");
      return;
    }
    const formData = new FormData();
    formData.append("file", selectedBatchFile);
    previewBatchButton.disabled = true;
    setBatchMessage("Validating CSV headers and every record…", "");
    try {
      const response = await fetch(API_BASE + "/validate/csv", {method:"POST", body:formData});
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = data.detail;
        throw new Error(typeof detail === "string" ? detail : detail?.message || JSON.stringify(detail));
      }
      batchValidated = true;
      document.getElementById("batch-preview-count").textContent =
        data.count + " valid records · " + data.columns + " required features";
      const previewFields = ["protocol_type","service","flag","duration","src_bytes","dst_bytes"];
      renderTable(
        document.getElementById("batch-preview-table"),
        ["Row", ...previewFields],
        data.preview.map((record, index) => [index + 1, ...previewFields.map(name => record[name])])
      );
      document.getElementById("batch-preview-wrap").classList.remove("hidden");
      runBatchButton.disabled = !(apiOnline && modelsReady);
      setBatchMessage("CSV validated successfully. Preview shown below. Ready to analyze " + data.count + " records.", "success");
    } catch (error) {
      batchValidated = false;
      setBatchMessage(error.message || "CSV validation failed.", "error");
    } finally {
      previewBatchButton.disabled = false;
    }
  });

  runBatchButton.addEventListener("click", async () => {
    if (!selectedBatchFile || !batchValidated || !apiOnline || !modelsReady) return;
    const formData = new FormData();
    formData.append("file", selectedBatchFile);
    const batchTask = document.getElementById("batch-mode").value;
    formData.append("task", batchTask);
    runBatchButton.disabled = true;
    previewBatchButton.disabled = true;
    setBatchMessage("Analyzing CSV records with the " + (batchTask === "binary_label" ? "binary" : "five-class") + " model…", "");
    try {
      const response = await fetch(API_BASE + "/predict/csv", {method:"POST", body:formData});
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = data.detail;
        throw new Error(typeof detail === "string" ? detail : detail?.message || JSON.stringify(detail));
      }
      batchPredictions = data.predictions;
      const counts = {};
      batchPredictions.forEach(item => { counts[item.prediction] = (counts[item.prediction] || 0) + 1; });
      const summary = document.getElementById("batch-summary");
      summary.replaceChildren();
      Object.entries(counts).sort((a,b) => b[1] - a[1]).forEach(([label, count]) => {
        const card = document.createElement("div");
        card.className = "batch-stat";
        const value = document.createElement("strong");
        value.textContent = count;
        const caption = document.createElement("span");
        caption.textContent = label;
        card.append(value, caption);
        summary.append(card);
      });
      const visibleRows = batchPredictions.slice(0, 100).map((item, index) => [
        index + 1,
        item.prediction,
        Number.isFinite(item.confidence) ? (item.confidence * 100).toFixed(1) + "%" : "N/A",
        typeof item.attack_probability === "number" ? (item.attack_probability * 100).toFixed(1) + "%" : "N/A"
      ]);
      renderTable(
        document.getElementById("batch-result-table"),
        ["Row", "Prediction", "Confidence", "Attack probability"],
        visibleRows
      );
      document.getElementById("batch-result-count").textContent =
        data.count + " predictions" + (data.count > 100 ? " · showing first 100" : "");
      document.getElementById("batch-results").classList.remove("hidden");
      setBatchMessage("Analysis complete. " + data.count + " records processed.", "success");
    } catch (error) {
      setBatchMessage(error.message || "Batch analysis failed.", "error");
    } finally {
      previewBatchButton.disabled = !selectedBatchFile;
      runBatchButton.disabled = !(batchValidated && apiOnline && modelsReady);
    }
  });

  document.getElementById("export-batch").addEventListener("click", () => {
    if (!batchPredictions.length) return;
    const headers = ["row", "prediction", "confidence", "attack_probability"];
    const escapeCsv = value => {
      const text = value == null ? "" : String(value);
      return '"' + text.replaceAll('"', '""') + '"';
    };
    const lines = [headers.join(",")];
    batchPredictions.forEach((item, index) => {
      lines.push([
        index + 1,
        item.prediction,
        item.confidence ?? "",
        item.attack_probability ?? ""
      ].map(escapeCsv).join(","));
    });
    const blob = new Blob([lines.join("\r\n")], {type:"text/csv;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "sentrinet_batch_predictions.csv";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });


  // Motion system: replay entrance, counter and chart animations whenever a view opens.
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  function animateCounter(element) {
    const finalText = element.dataset.finalText || element.textContent.trim();
    element.dataset.finalText = finalText;
    const match = finalText.match(/^([^0-9-]*)(-?\\d[\\d,]*(?:\\.\\d+)?)(.*)$/);
    if (!match) return;
    const prefix = match[1], raw = match[2], suffix = match[3];
    const target = Number(raw.replace(/,/g, ""));
    if (!Number.isFinite(target)) return;
    const decimals = (raw.split(".")[1] || "").length;
    const padWidth = raw.replace(/,/g, "").split(".")[0].length;
    if (reduceMotion) { element.textContent = finalText; return; }
    const duration = 1050, start = performance.now();
    element.textContent = prefix + (0).toFixed(decimals) + suffix;
    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 4);
      let formatted = (target * eased).toLocaleString("en-US", {minimumFractionDigits:decimals, maximumFractionDigits:decimals});
      if (/^0\\d/.test(raw) && target < 10) formatted = formatted.replace(/^\\d+/, value => value.padStart(padWidth, "0"));
      element.textContent = prefix + formatted + suffix;
      if (progress < 1) requestAnimationFrame(tick); else element.textContent = finalText;
    }
    requestAnimationFrame(tick);
  }

  function animateView(view) {
    if (!view) return;
    view.classList.remove("motion-enter");
    void view.offsetWidth;
    view.classList.add("motion-enter");
    if (reduceMotion) return;
    view.querySelectorAll(".metric-card strong, .overview-stat strong").forEach(animateCounter);
    view.querySelectorAll(".bar-fill").forEach(bar => {
      const targetWidth = bar.dataset.targetWidth || bar.style.width;
      bar.dataset.targetWidth = targetWidth;
      bar.style.width = "0";
      requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = targetWidth; }));
    });
  }

  // Sidebar navigation switches independent workspace views instead of scrolling anchors.
  const viewTitles = {overview:"Overview",analyzer:"Threat analyzer",batch:"Batch analysis",performance:"Model performance",models:"Model information"};
  function showView(name) {
    if (!viewTitles[name]) return;
    document.querySelectorAll(".app-view").forEach(view => view.classList.toggle("active", view.dataset.view === name));
    document.querySelectorAll(".nav-item[data-nav]").forEach(link => {
      const active = link.dataset.nav === name;
      link.classList.toggle("active", active);
      const dot = link.querySelector(".nav-dot");
      if (active && !dot) {
        const marker = document.createElement("span");
        marker.className = "nav-dot";
        link.append(marker);
      } else if (!active && dot) dot.remove();
    });
    document.getElementById("breadcrumb-current").textContent = viewTitles[name];
    animateView(document.querySelector('.app-view[data-view="' + name + '"]'));
    history.replaceState(null, "", "#" + name);
    window.scrollTo({top:0,behavior:"auto"});
  }
  document.querySelectorAll(".nav-item[data-nav]").forEach(link => {
    link.addEventListener("click", event => {
      event.preventDefault();
      showView(link.dataset.nav);
    });
  });
  document.querySelectorAll("[data-open-view]").forEach(link => {
    link.addEventListener("click", event => {
      event.preventDefault();
      showView(link.dataset.openView);
    });
  });
  const initialView = location.hash.replace("#","");
  showView(viewTitles[initialView] ? initialView : "overview");

  renderForm();
  if (allFields.length !== 40) {
    showError("Frontend schema mismatch: expected 40 features but rendered " + allFields.length + ".");
  }
  refreshHealth();
})();
