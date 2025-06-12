# Supply Chain Risk Tracker - Frontend

A modern React dashboard for monitoring and predicting supply chain risks using AI/ML.

## Features

- **Real-time Dashboard** - Live metrics and system status
- **Company Management** - Monitor companies and their risk profiles
- **Risk Predictions** - AI-powered risk forecasting with 45-day advance warning
- **Analytics** - Advanced charts and insights
- **System Monitoring** - Health checks and data quality metrics

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **React Query** for data fetching and caching
- **React Router** for navigation
- **Recharts** for data visualization
- **Heroicons** for icons

## Quick Start

1. **Install dependencies:**

   ```bash
   npm install
   ```

2. **Start development server:**

   ```bash
   npm run dev
   ```

3. **Open browser:**
   Navigate to `http://localhost:3000`

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## API Integration

The frontend connects to the Go API server at `http://localhost:8080/api/v1`. Make sure the backend is running before starting the frontend.

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── Layout/         # Layout components (Header, Sidebar)
│   ├── Charts/         # Chart components
│   └── ...
├── pages/              # Page components
├── services/           # API service layer
├── types/              # TypeScript type definitions
└── main.tsx           # Application entry point
```

## Environment Variables

The frontend uses Vite's proxy configuration to connect to the backend API. No additional environment variables are required for development.

## Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.
