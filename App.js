import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import SplashPage from "./SplashPage";
import UploadPage from "./UploadPage";
import ResultsPage from "./ResultsPage"; // Import ResultsPage

function App() {
  return (
    <Router>
      <Routes>
        {/* Define the routes for different pages */}
        <Route path="/" element={<SplashPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/results" element={<ResultsPage />} /> {/* Render ResultsPage */}
      </Routes>
    </Router>
  );
}

export default App;