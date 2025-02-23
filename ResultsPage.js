
import { motion } from "framer-motion";

export default function ResultsPage() {
  

  // Define the path for the local video if you want to play a video from your local directory
  const videoPath = "/Users/nikhilkasam/Downloads/IMG_5953.mp4"; // Ensure this is the correct relative path from public

  return (
    <div className="min-h-screen bg-black text-white flex flex-col items-center justify-center relative">
      {/* Background gradient effect */}
      <motion.div
        className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-cyan-600 to-purple-600 opacity-30"
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.3 }}
        transition={{ duration: 1.5, ease: "easeOut" }}
      ></motion.div>

      {/* Title */}
      <motion.h1
        className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500 mb-6 z-10"
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        🎥 Your Form Analysis
      </motion.h1>

      {/* Layout */}
      <div className="flex w-full max-w-screen-xl mx-auto justify-between gap-12 px-4">
        {/* Left side - Video */}
        <motion.div
          className="flex-1 bg-gray-800 rounded-lg p-4"
          initial={{ opacity: 0, x: -100 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 1, ease: "easeOut" }}
        >
          {/* Video element */}
          <video
            controls
            className="w-full rounded-lg"
            autoPlay
            muted
            // Use the path for local video here
          >
            <source src={videoPath} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </motion.div>

        {/* Right side - Recommendations */}
        <motion.div
          className="flex-1 bg-gray-800 rounded-lg p-8"
          initial={{ opacity: 0, x: 100 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 1, ease: "easeOut" }}
        >
          <h2 className="text-2xl font-bold text-center mb-6">Recommendations</h2>
          <p className="text-lg">
            Based on the analysis, here are some suggestions for improving your form:
          </p>
          <ul className="list-disc pl-6 mt-4">
            <li>Keep your back straight to avoid strain.</li>
            <li>Engage your core during every rep.</li>
            <li>Slow down the movements for better muscle engagement.</li>
            <li>Avoid locking your knees during squats.</li>
          </ul>
        </motion.div>
      </div>
    </div>
  );
}