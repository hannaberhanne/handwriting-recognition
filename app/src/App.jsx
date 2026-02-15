import { useState } from 'react';

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export default function App() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [filename, setFilename] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError('Choose an image to upload.');
      return;
    }

    setLoading(true);
    setError('');

    const payload = new FormData();
    payload.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: payload,
      });

      if (!response.ok) {
        throw new Error('Server returned an error');
      }

      const result = await response.json();
      setFilename(result.filename ?? '');
      setPredictions(result.predictions ?? []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <h1>Ghana OCR Upload</h1>
      <p>Upload a snapshot and see stubbed OCR predictions.</p>

      <form className="upload-form" onSubmit={handleSubmit}>
        <label className="file-input">
          <span>Choose image</span>
          <input
            type="file"
            accept="image/*"
            onChange={(event) => {
              setFile(event.target.files?.[0] ?? null);
              setError('');
            }}
          />
        </label>

        <button type="submit" disabled={loading}>
          {loading ? 'Uploading…' : 'Submit'}
        </button>
      </form>

      {error && <p className="callout error">{error}</p>}

      {filename && (
        <section className="results">
          <h2>Results for {filename}</h2>
          <ul>
            {predictions.map((pred, index) => (
              <li key={`${pred.label}-${index}`}>
                <strong>{pred.label}</strong>
                <span>{(pred.confidence * 100).toFixed(0)}% confidence</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}
