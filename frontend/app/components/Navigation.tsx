'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Menu, X, Brain } from 'lucide-react';
import ModelSelector from './ModelSelector';

export default function Navigation() {
    const [isOpen, setIsOpen] = useState(false);
    const pathname = usePathname();

    const links = [
        { href: '/asd-detection', label: 'ASD Detection' },
        { href: '/emotion-recognition', label: 'Emotion Recognition' },
        { href: '/combined-analysis', label: 'Combined Analysis' },
        { href: '/gamification', label: 'Gamification' },
    ];

    const isActive = (path: string) => pathname === path;

    return (
        <nav className="bg-white shadow-lg sticky top-0 z-50">
            <div className="w-full mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16">
                    {/* Logo and Desktop Links */}
                    <div className="flex items-center">
                        <Link href="/" className="flex-shrink-0 flex items-center gap-2 text-xl md:text-2xl font-black text-blue-600 truncate mr-6">
                            <Brain className="w-6 h-6 md:w-8 md:h-8" />
                            ASD-Gamiscreen
                        </Link>

                        {/* Desktop Menu */}
                        <div className="hidden lg:ml-4 lg:flex lg:space-x-3 xl:space-x-6">
                            {links.map(link => (
                                <Link
                                    key={link.href}
                                    href={link.href}
                                    className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm xl:text-base font-medium whitespace-nowrap transition-colors ${isActive(link.href)
                                        ? 'border-blue-500 text-blue-600'
                                        : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                                        }`}
                                >
                                    {link.label}
                                </Link>
                            ))}
                        </div>
                    </div>

                    {/* Right side: Model Selector + Mobile Toggle */}
                    <div className="flex items-center gap-2 sm:gap-4 ml-4 lg:ml-8 flex-shrink-0">
                        <div className="hidden sm:block">
                            <ModelSelector />
                        </div>

                        {/* Mobile menu button */}
                        <div className="flex items-center lg:hidden ml-2">
                            <button
                                onClick={() => setIsOpen(!isOpen)}
                                className="inline-flex items-center justify-center p-2 rounded-md text-gray-500 hover:text-gray-700 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500 transition-colors"
                            >
                                <span className="sr-only">Open main menu</span>
                                {isOpen ? (
                                    <X className="block h-6 w-6" aria-hidden="true" />
                                ) : (
                                    <Menu className="block h-6 w-6" aria-hidden="true" />
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Mobile Menu Panel */}
            {isOpen && (
                <div className="lg:hidden bg-white border-t border-gray-100 shadow-xl absolute w-full left-0 animate-fade-in">
                    <div className="px-4 pt-2 pb-3 space-y-1">
                        {links.map(link => (
                            <Link
                                key={link.href}
                                href={link.href}
                                onClick={() => setIsOpen(false)}
                                className={`block pl-3 pr-4 py-3 border-l-4 text-base font-medium transition-colors ${isActive(link.href)
                                    ? 'bg-blue-50 border-blue-500 text-blue-700'
                                    : 'border-transparent text-gray-600 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-800'
                                    }`}
                            >
                                {link.label}
                            </Link>
                        ))}

                        {/* Mobile Model Selector */}
                        <div className="pl-3 pr-4 py-4 mt-4 border-t border-gray-100 sm:hidden">
                            <p className="text-sm text-gray-500 mb-2 font-medium">Active Model:</p>
                            <ModelSelector />
                        </div>
                    </div>
                </div>
            )}
        </nav>
    );
}
