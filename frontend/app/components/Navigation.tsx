'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Menu, X, Brain, Smile, Layers, Gamepad2, Home } from 'lucide-react';

export default function Navigation() {
    const [isOpen, setIsOpen] = useState(false);
    const pathname = usePathname();

    const links = [
        { href: '/', label: 'Home', icon: Home },
        { href: '/asd-detection', label: 'Screening', icon: Brain },
        { href: '/emotion-recognition', label: 'Emotions', icon: Smile },
        { href: '/combined-analysis', label: 'Analysis', icon: Layers },
        { href: '/gamification', label: 'Learning', icon: Gamepad2 },
    ];

    const isActive = (path: string) => pathname === path;

    return (
        <nav className="bg-white/80 backdrop-blur-md shadow-sm sticky top-0 z-50 border-b border-gray-100">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-18 py-3">
                    {/* Logo */}
                    <div className="flex items-center">
                        <Link href="/" className="flex items-center gap-2 text-2xl font-black text-blue-600 hover:opacity-80 transition-opacity">
                            <div className="p-1.5 bg-blue-600 rounded-lg shadow-blue-200 shadow-lg">
                                <Brain className="w-6 h-6 text-white" />
                            </div>
                            <span className="hidden lg:inline-block tracking-tighter">ASD-Gamiscreen</span>
                        </Link>
                    </div>

                    {/* Desktop Menu */}
                    <div className="hidden lg:flex items-center bg-gray-100/50 p-1 rounded-2xl border border-gray-200/50">
                        {links.map(link => {
                            const Icon = link.icon;
                            return (
                                <Link
                                    key={link.href}
                                    href={link.href}
                                    className={`flex items-center gap-1.5 px-3 py-2 rounded-xl text-[13px] font-bold transition-all duration-200 ${isActive(link.href)
                                        ? 'bg-white text-blue-600 shadow-sm border border-gray-100'
                                        : 'text-gray-500 hover:text-gray-900 hover:bg-white/50'
                                    }`}
                                >
                                    <Icon className={`w-3.5 h-3.5 ${isActive(link.href) ? 'text-blue-600' : 'text-gray-400'}`} />
                                    {link.label}
                                </Link>
                            );
                        })}
                    </div>

                    {/* Right side Actions (e.g., Get Started) */}
                    <div className="flex items-center gap-3">
                        <Link 
                            href="/asd-detection" 
                            className="hidden md:flex btn bg-blue-600 text-white hover:bg-blue-700 shadow-md shadow-blue-200 px-6 py-2 rounded-xl text-sm font-bold"
                        >
                            Get Started
                        </Link>

                        {/* Mobile menu button */}
                        <button
                            onClick={() => setIsOpen(!isOpen)}
                            className="lg:hidden p-2 rounded-xl text-gray-500 hover:bg-gray-100 border border-transparent hover:border-gray-200 transition-all"
                        >
                            {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                        </button>
                    </div>
                </div>
            </div>

            {/* Mobile Menu */}
            {isOpen && (
                <div className="lg:hidden bg-white/95 backdrop-blur-xl border-t border-gray-100 shadow-2xl absolute w-full left-0 animate-slide-down duration-300">
                    <div className="px-4 py-6 space-y-2">
                        {links.map(link => {
                            const Icon = link.icon;
                            return (
                                <Link
                                    key={link.href}
                                    href={link.href}
                                    onClick={() => setIsOpen(false)}
                                    className={`flex items-center gap-4 px-4 py-4 rounded-2xl text-base font-bold transition-all ${isActive(link.href)
                                        ? 'bg-blue-50 text-blue-700 border border-blue-100'
                                        : 'text-gray-600 hover:bg-gray-50'
                                    }`}
                                >
                                    <div className={`p-2 rounded-xl ${isActive(link.href) ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-400'}`}>
                                        <Icon className="w-5 h-5" />
                                    </div>
                                    {link.label}
                                </Link>
                            );
                        })}
                        <div className="pt-4 mt-4 border-t border-gray-100">
                             <Link 
                                href="/asd-detection" 
                                onClick={() => setIsOpen(false)}
                                className="flex items-center justify-center p-4 rounded-2xl bg-blue-600 text-white font-bold"
                            >
                                Start Screening
                            </Link>
                        </div>
                    </div>
                </div>
            )}
        </nav>
    );
}
