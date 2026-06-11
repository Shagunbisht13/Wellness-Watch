import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5000/api',
});

export const getLatestPrediction = async () => {
  try {
    const res = await api.get('/latest');
    return res.data;
  } catch (err) {
    console.error("Error fetching latest prediction:", err);
    return null;
  }
};

export const getHistory = async () => {
  try {
    const res = await api.get('/history');
    return res.data;
  } catch (err) {
    console.error("Error fetching history:", err);
    return [];
  }
};

export const getStats = async () => {
  try {
    const res = await api.get('/stats');
    return res.data;
  } catch (err) {
    console.error("Error fetching stats:", err);
    return {};
  }
};

export const getProofUrl = (path) => {
  if (!path) return null;
  // path is something like "proof/images/image_xxx.jpg"
  // The API serves it from /proof/...
  // We need to strip the leading "proof/" since our endpoint is /proof/<filename>
  const filename = path.replace(/^proof\//, '');
  return `http://localhost:5000/proof/${filename}`;
};
