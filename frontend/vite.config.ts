import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    target: "es2021",
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/three")) {
            return "three-core";
          }
          if (
            id.includes("@react-three") ||
            id.includes("three-stdlib") ||
            id.includes("meshline") ||
            id.includes("@use-gesture")
          ) {
            return "three-helpers";
          }
          if (id.includes("node_modules/react") || id.includes("node_modules/scheduler")) {
            return "react-vendor";
          }
          if (id.includes("node_modules/zustand")) {
            return "state-vendor";
          }
          return undefined;
        },
      },
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
  },
  preview: {
    host: "0.0.0.0",
    port: 4173,
  },
});
