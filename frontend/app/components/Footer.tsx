import Link from 'next/link';
import { Brain, Heart, Github, Twitter, Mail } from 'lucide-react';

export default function Footer() {
    const currentYear = new Date().getFullYear();

    return (
        <footer className="bg-white border-t border-gray-100 pb-8 pt-16 mt-auto">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-12">
                    {/* Brand Section */}
                    <div className="md:col-span-2 space-y-4">
                        <Link href="/" className="inline-flex items-center gap-2 text-2xl font-black text-blue-600">
                            <Brain className="w-8 h-8 text-blue-600" />
                            ASD-Gamiscreen
                        </Link>
                        <p className="text-gray-500 max-w-sm text-sm leading-relaxed">
                            Advanced AI-powered system for autism spectrum disorder screening and real-time emotion analysis with explainable AI visualizations.
                        </p>
                        <div className="flex gap-4 pt-4">
                            <a href="#" className="p-2 bg-gray-50 text-gray-500 rounded-full hover:bg-blue-50 hover:text-blue-600 transition-colors">
                                <Twitter className="w-5 h-5" />
                            </a>
                            <a href="#" className="p-2 bg-gray-50 text-gray-500 rounded-full hover:bg-blue-50 hover:text-blue-600 transition-colors">
                                <Github className="w-5 h-5" />
                            </a>
                            <a href="#" className="p-2 bg-gray-50 text-gray-500 rounded-full hover:bg-blue-50 hover:text-blue-600 transition-colors">
                                <Mail className="w-5 h-5" />
                            </a>
                        </div>
                    </div>

                    {/* Features Links */}
                    <div className="space-y-4">
                        <h4 className="font-bold text-gray-900 group flex items-center gap-2">
                            Features
                        </h4>
                        <ul className="space-y-3">
                            <li>
                                <Link href="/asd-detection" className="text-gray-500 hover:text-blue-600 text-sm font-medium transition-colors">
                                    ASD Detection
                                </Link>
                            </li>
                            <li>
                                <Link href="/emotion-recognition" className="text-gray-500 hover:text-blue-600 text-sm font-medium transition-colors">
                                    Emotion Recognition
                                </Link>
                            </li>
                            <li>
                                <Link href="/combined-analysis" className="text-gray-500 hover:text-blue-600 text-sm font-medium transition-colors">
                                    Combined Analysis
                                </Link>
                            </li>
                            <li>
                                <Link href="/gamification" className="text-gray-500 hover:text-blue-600 text-sm font-medium transition-colors flex items-center gap-2">
                                    Gamification <span className="px-2 py-0.5 rounded-full bg-yellow-100 text-yellow-700 text-[10px] uppercase font-bold tracking-wider">New</span>
                                </Link>
                            </li>
                        </ul>
                    </div>

                    {/* About Links */}
                    <div className="space-y-4">
                        <h4 className="font-bold text-gray-900">
                            Resources
                        </h4>
                        <ul className="space-y-3">
                            <li>
                                <a href="#" className="text-gray-500 hover:text-blue-600 text-sm font-medium transition-colors">
                                    Documentation
                                </a>
                            </li>
                            <li>
                                <a href="#" className="text-gray-500 hover:text-blue-600 text-sm font-medium transition-colors">
                                    Privacy Policy
                                </a>
                            </li>
                            <li>
                                <a href="#" className="text-gray-500 hover:text-blue-600 text-sm font-medium transition-colors">
                                    Terms of Service
                                </a>
                            </li>
                        </ul>
                    </div>
                </div>

                {/* Bottom Bar */}
                <div className="pt-8 border-t border-gray-100 flex flex-col md:flex-row justify-between items-center gap-4">
                    <p className="text-sm font-medium text-gray-400">
                        &copy; {currentYear} ASD-Gamiscreen. All rights reserved.
                    </p>
                    <p className="text-sm font-medium text-gray-400 flex items-center gap-1">
                        Made with <Heart className="w-4 h-4 text-red-500 animate-pulse fill-red-500" /> for Autism Support
                    </p>
                </div>
            </div>
        </footer>
    );
}
