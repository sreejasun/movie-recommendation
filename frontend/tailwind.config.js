/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          400: "#ef4444",
          500: "#dc2626",
          600: "#b91c1c"
        }
      },
      boxShadow: {
        glass: "0 10px 35px rgba(2, 6, 23, 0.45)"
      },
      animation: {
        floatIn: "floatIn 0.4s ease-out"
      },
      keyframes: {
        floatIn: {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" }
        }
      }
    }
  },
  plugins: []
};
