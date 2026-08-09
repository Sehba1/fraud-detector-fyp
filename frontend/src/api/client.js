import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000,
});

export async function analyseText(text) {
  const response = await client.post("/predict/text", { text });
  return response.data;
}

export async function analyseImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await client.post("/predict/image", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
}

export async function checkHealth() {
  const response = await client.get("/health");
  return response.data;
}