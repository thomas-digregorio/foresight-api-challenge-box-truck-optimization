import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      boxShadow: {
        panel: "0 24px 80px rgba(9, 15, 22, 0.38)",
      },
      colors: {
        ink: "#0f1723",
        panel: "#121a26",
        glow: "#dca13b",
        wall: "#f3efe7",
        wood: "#b88757",
      },
      borderRadius: {
        "4xl": "2rem",
      },
    },
  },
  plugins: [],
};

export default config;

