import React, { useState } from "react";
import { fadeIn, slideInFromBottom, slideInFromLeft, slideInFromRight, slideInFromTop, staggeredChildren } from "../lib/animations";
import "../App.css"

const Predictor: React.FC = () => {
  const getStaggered = staggeredChildren(fadeIn, 100, 100);
  const [formData, setFormData] = useState({
    C1R1: "",
    C1G1: "",
    C1B1: "",
    PH1: "",
    Fe_plus2: "",
    Fe_plus3: "",
    acid_conc: "",
    pulp_density: "",
    temp: "",
    time: "",
  });

  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    const data = Object.fromEntries(
      Object.entries(formData).map(([key, value]) => [key, parseFloat(value)])
    );

    console.log(data);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });
      console.log(response);

      const json = await response.json();
      if ("Copper_Recovery" in json) {
        setResult(`✅ Predicted Copper Recovery: ${(json.Copper_Recovery * 100).toFixed(2)}%`);
      } else {
        setResult("❌ Prediction failed");
      }
    } catch (error) {
      console.error(error);
      setResult("❌ Error: Unable to get prediction");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <div className={`flex justify-center items-center mt-5 ${slideInFromTop}`}>
        <h1 className="page-title">PCB Copper Extraction Predictor</h1>
      </div>
      <form onSubmit={handleSubmit} className={`${slideInFromBottom}`} style={styles.form}>
        {Object.keys(formData).map((key) => (
            <input
              key={key}
              type="number"
              step="any"
              name={key}
              placeholder={key}
              value={(formData as any)[key]}
              onChange={handleChange}
              required
              style={styles.input}
              className={`dark:bg-eco-green-dark rounded-lg shadow-sm ${slideInFromLeft} hover:shadow-lg transition-all duration-150`}
            />
        ))}
        <button className="button" type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict Recovery"}
        </button>
      </form>
      {result && <div style={styles.result}>{result}</div>}
    </div>
  );
};

const styles: { [key: string]: React.CSSProperties } = {
  container: {
    fontFamily: "Arial, sans-serif",
    padding: 20,
    background: "#f4f4f4",
    minHeight: "100vh",
  },
  form: {
    padding: 20,
    borderRadius: 10,
    maxWidth: 600,
    margin: "auto",
    display: "flex",
    flexDirection: "column",
  },
  input: {
    padding: 8,
    paddingLeft: 20,
    margin: "8px 0",
    fontSize: 16,
    height: 50,
    borderRadius: 30,
    borderWidth: 1,
    borderColor: 'green',
  }, 
  button: {
    backgroundColor: "#27ae60",
    color: "white",
    padding: "10px 20px",
    border: "none",
    fontSize: 16,
    fontWeight: "bold",
    marginTop: 10,
    cursor: "pointer",
    borderRadius: 30
  },
  result: {
    marginTop: 20,
    fontSize: 18,
    color: "#2980b9",
    textAlign: "center",
  },
};

export default Predictor;
