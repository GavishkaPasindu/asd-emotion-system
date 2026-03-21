import type { Metadata } from "next";
import "./globals.css";
import Navigation from "./components/Navigation";

export const metadata: Metadata = {
  title: "ASD-Gamiscreen - Advanced Behavioral Analysis",
  description: "AI-powered developmental screening and emotion recognition with explainable AI visualizations",
};

import { ModelProvider } from "./context/ModelContext";
import Footer from "./components/Footer";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen flex flex-col">
        <ModelProvider>
          <Navigation />
          {children}
          <Footer />
        </ModelProvider>
      </body>
    </html>
  );
}
