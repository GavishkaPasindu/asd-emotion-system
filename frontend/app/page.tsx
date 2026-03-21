import Link from 'next/link';
import { Brain, Smile, Layers, Activity, BarChart3, Sparkles, Shield, Zap } from 'lucide-react';

export default function Home() {
  const features = [
    {
      icon: Brain,
      title: 'Developmental Screening',
      description: 'Advanced AI model for child development condition screening with high accuracy',
      href: '/asd-detection',
      gradient: 'from-purple-500 to-blue-500',
    },
    {
      icon: Smile,
      title: 'Emotion Recognition',
      description: 'Real-time facial emotion analysis with detailed probability distributions',
      href: '/emotion-recognition',
      gradient: 'from-pink-500 to-purple-500',
    },
    {
      icon: Layers,
      title: 'Combined Analysis',
      description: 'Comprehensive assessment combining behavioral screening and emotion recognition',
      href: '/combined-analysis',
      gradient: 'from-blue-500 to-cyan-500',
    },
    {
      icon: Activity,
      title: 'Real-time Tracking',
      description: 'Live emotion monitoring and continuous facial analysis',
      href: '#',
      gradient: 'from-green-500 to-emerald-500',
    },
    {
      icon: Sparkles,
      title: 'Explainable AI',
      description: 'Grad-CAM heatmaps showing exactly what the AI focuses on',
      href: '/asd-detection',
      gradient: 'from-violet-500 to-purple-500',
    },
    {
      icon: Smile, // Reusing Smile for now, or import Gamepad/Joystick if available
      title: 'Interactive Support',
      description: 'Fun, child-friendly support with cartoons and interactive elements',
      href: '/gamification',
      gradient: 'from-yellow-400 to-orange-500',
    },
  ];

  return (
    <main className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden gradient-mesh py-20 md:py-32">
        <div className="container-custom relative z-10">
          <div className="max-w-4xl mx-auto text-center animate-fade-in">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 mb-6">
              <Shield className="w-4 h-4 text-[rgb(var(--color-primary))]" />
              <span className="text-sm font-medium">Powered by Advanced AI</span>
            </div>

            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              <span className="text-gradient">ASD-Gamiscreen</span>
            </h1>

            <p className="text-xl md:text-2xl text-[rgb(var(--color-text-secondary))] mb-8 max-w-2xl mx-auto">
              Advanced AI-powered system for behavioral screening and real-time emotion analysis with explainable AI visualizations
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mt-8">
              <Link href="/asd-detection" className="flex items-center justify-center px-8 py-4 rounded-full bg-blue-600 text-white font-bold text-lg hover:bg-blue-700 hover:shadow-lg hover:-translate-y-1 transition-all">
                <Brain className="w-5 h-5 mr-3" />
                Begin Screening
              </Link>
              <Link href="/emotion-recognition" className="flex items-center justify-center px-8 py-4 rounded-full bg-white text-gray-800 font-bold text-lg border-2 border-gray-200 hover:border-gray-300 hover:bg-gray-50 hover:shadow-lg hover:-translate-y-1 transition-all">
                <Smile className="w-5 h-5 mr-3" />
                Try Emotion Recognition
              </Link>
            </div>
          </div>
        </div>

        {/* Animated background elements */}
        <div className="absolute top-20 left-10 w-72 h-72 bg-[rgb(var(--color-primary)/0.1)] rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-[rgb(var(--color-secondary)/0.1)] rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
      </section>

      {/* Features Grid */}
      <section className="py-20 bg-[rgb(var(--color-bg))]">
        <div className="container-custom">
          <div className="text-center mb-16 animate-slide-up">
            <h2 className="text-4xl md:text-5xl font-bold mb-4 text-[rgb(var(--color-text))]">
              Powerful Features
            </h2>
            <p className="text-xl text-[rgb(var(--color-text-secondary))] max-w-2xl mx-auto">
              Everything you need for comprehensive emotion and behavioral analysis
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <Link
                  key={index}
                  href={feature.href}
                  className="card hover-lift group"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <div className={`p-3 rounded-xl bg-gradient-to-br ${feature.gradient} w-fit mb-4 group-hover:scale-110 transition-transform duration-300`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>

                  <h3 className="text-xl font-semibold mb-2 text-[rgb(var(--color-text))]">
                    {feature.title}
                  </h3>

                  <p className="text-[rgb(var(--color-text-secondary))]">
                    {feature.description}
                  </p>

                  <div className="mt-4 flex items-center gap-2 text-[rgb(var(--color-primary))] font-medium text-sm group-hover:gap-3 transition-all">
                    Learn more
                    <Zap className="w-4 h-4" />
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 gradient-primary">
        <div className="container-custom text-center">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Ready to Get Started?
          </h2>
          <p className="text-xl text-white/90 mb-8 max-w-2xl mx-auto">
            Experience the power of AI-driven emotion and behavioral analysis
          </p>
          <Link href="/combined-analysis" className="btn bg-white text-[rgb(var(--color-primary))] hover:bg-white/90">
            <Layers className="w-5 h-5 inline mr-2" />
            Try Combined Analysis
          </Link>
        </div>
      </section>
    </main>
  );
}
