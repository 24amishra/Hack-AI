import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import osulogo from './Ohio-State-Logo.png'; // Import the image

export default function SplashPage() {
  return (
    <div className="min-h-screen bg-black text-white flex flex-col items-center pt-12 relative">
      {/* Background image with motion */}
      <motion.div 
        className="absolute top-6 left-6 w-20 h-20 opacity-50"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
      >
        <img src={osulogo} alt="OSU logo" className="w-full h-full object-contain" loading="lazy" />
      </motion.div>

      {/* Title and description with futuristic font */}
      <motion.div
        className="text-center mb-8 z-10"
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <h1 className="text-5xl font-extrabold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-600">
          🏋️ LiftSense.AI
        </h1>
        <p className="text-xl mb-6 font-light tracking-wide">
          Precision fitness with AI-driven form analysis.
        </p>
      </motion.div>

      {/* Product Description with smooth fade-in */}
      <motion.div
        className="text-center mb-8 px-4 z-10"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1.2, ease: "easeInOut" }}
      >
        <p className="text-lg mb-4 max-w-xl mx-auto">
          LiftSense.AI uses cutting-edge MediaPipe Pose technology to analyze your form, ensuring your exercises are performed with precision. Our system helps you minimize injury risks while maximizing performance.

          The process is simple and fast: upload a video of your workout, and LiftSense.AI provides instant feedback, highlighting areas for improvement and providing actionable insights to refine your technique.

          Whether you’re just starting your fitness journey or looking to fine-tune your skills, LiftSense.AI makes form correction easy and accessible, no personal trainer required.
        </p>
      </motion.div>

      {/* Call to Action Button with futuristic hover effect */}
      <Link to="/upload">
        <button className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg shadow-lg transform transition-all duration-300 hover:scale-105 z-10">
          📤 Analyze Your Form Now
        </button>
      </Link>

      {/* Optional Decorative Line */}
      <motion.div
        className="absolute bottom-0 w-full bg-gradient-to-r from-cyan-400 to-purple-600 h-2 z-0"
        initial={{ width: 0 }}
        animate={{ width: "100%" }}
        transition={{ duration: 1.2, ease: "easeInOut" }}
      ></motion.div>
    </div>
  );
}