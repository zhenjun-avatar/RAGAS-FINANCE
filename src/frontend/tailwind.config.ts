import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#f4f4f5",
        foreground: "#18181b",
        card: "#ffffff",
        border: "#e4e4e7",
        muted: "#71717a"
      }
    }
  },
  plugins: []
};

export default config;
