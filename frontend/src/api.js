const API_BASE =
  import.meta.env.VITE_API_BASE ||
  "https://crack-detection-shm.onrender.com";

if (import.meta.env.DEV) {
  // Helps catch the "silently hitting production" bug: confirms which backend the app is calling.
  console.info("[api] using base:", API_BASE);
}

async function request(path) {
  const response = await fetch(`${API_BASE}${path}`);

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${response.status}`);
  }

  return response.json();
}

export function imageUrl(imageId) {
  return `${API_BASE}/api/predictions/${encodeURIComponent(imageId)}/image`;
}

export function artifactUrl(imageId, artifact) {
  return `${API_BASE}/api/predictions/${encodeURIComponent(imageId)}/${artifact}`;
}

export function projectArtifactUrl(projectId, imageId, artifact) {
  return `${API_BASE}/api/projects/${encodeURIComponent(projectId)}/images/${encodeURIComponent(imageId)}/${artifact}`;
}

export function getSummary() {
  return request("/api/summary");
}

export function getMethodology() {
  return request("/api/methodology");
}

export function getOptions() {
  return request("/api/options");
}

export function getPredictions(filters) {
  const params = new URLSearchParams();

  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      params.set(key, value);
    }
  });

  return request(`/api/predictions?${params.toString()}`);
}

export async function uploadProject(name, files, scaleMmPerPx) {
  const formData = new FormData();

  formData.set("name", name || "Concrete Inspection Project");

  if (scaleMmPerPx !== undefined && scaleMmPerPx !== null && scaleMmPerPx !== "") {
    formData.set("scale_mm_per_px", String(scaleMmPerPx));
  }

  Array.from(files).forEach((file) => {
    formData.append("files", file);
  });

  const response = await fetch(`${API_BASE}/api/projects`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Upload failed: ${response.status}`);
  }

  return response.json();
}