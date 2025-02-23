import { useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

export default function UploadPage() {
  const [file, setFile] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  return (
    <div className="min-h-screen bg-black text-white flex flex-col items-center justify-center relative">
      {/* Background gradient effect */}
      <motion.div
        className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-cyan-600 to-purple-600 opacity-30"
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.3 }}
        transition={{ duration: 1.5, ease: "easeOut" }}
      ></motion.div>

      {/* Page Title */}
      <motion.h1
        className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500 mb-6 z-10"
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        🎥 Form Analysis
      </motion.h1>

      {/* Description */}
      <motion.p
        className="text-lg text-center mb-8 z-10"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1, ease: "easeInOut" }}
      >
        Upload a video of your exercise, and let LiftSense.AI analyze your form.
      </motion.p>

      {/* File Upload Section */}
      <motion.label
        className="border-2 border-dashed border-gray-500 p-8 rounded-lg cursor-pointer transform transition-all duration-300 hover:scale-105 hover:border-cyan-500 z-10"
        initial={{ scale: 0.95 }}
        animate={{ scale: 1 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
      >
        <input
          type="file"
          className="hidden"
          accept="video/*"
          onChange={handleFileChange}
        />
        {file ? (
          <p className="text-center text-white">{file.name}</p>
        ) : (
          <div className="text-center text-white">
            📂 Click or drag a video to upload
          </div>
        )}
      </motion.label>

      {/* Action Button */}
      <motion.div
        className="mt-8 z-20" // Increased z-index here to ensure the button is clickable
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1.2, ease: "easeInOut" }}
      >
        {/* Link to the ResultsPage, passing file state */}
        <Link
          to={{
            pathname: "/results", // Navigate to ResultsPage
            state: { file }, // Pass the file as state
          }}
        >
          <motion.button
            className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-400 hover:to-purple-400 text-white px-8 py-4 rounded-lg shadow-lg transform transition-all duration-300 hover:scale-105 z-30" // Higher z-index for the button
            whileHover={{ scale: 1.1 }}
            transition={{ type: "spring", stiffness: 400, damping: 15 }}
          >
            Analyze Now
          </motion.button>
        </Link>
      </motion.div>
    </div>
  );
}