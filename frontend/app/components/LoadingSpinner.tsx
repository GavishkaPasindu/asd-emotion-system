'use client';

export default function LoadingSpinner({ size = 'md', text }: { size?: 'sm' | 'md' | 'lg'; text?: string }) {
    const sizeClasses = {
        sm: 'w-8 h-8',
        md: 'w-12 h-12',
        lg: 'w-16 h-16',
    };

    return (
        <div className="flex flex-col items-center justify-center gap-4 p-8">
            <div className="relative">
                {/* Outer ring */}
                <div className={`${sizeClasses[size]} rounded-full border-4 border-[rgb(var(--color-bg-secondary))]`} />

                {/* Spinning gradient ring */}
                <div
                    className={`
            ${sizeClasses[size]} rounded-full absolute inset-0
            border-4 border-transparent border-t-[rgb(var(--color-primary))] 
            border-r-[rgb(var(--color-secondary))]
            animate-spin
          `}
                    style={{ animationDuration: '1s' }}
                />

                {/* Inner glow */}
                <div
                    className={`
            ${sizeClasses[size]} rounded-full absolute inset-0
            bg-gradient-to-br from-[rgb(var(--color-primary)/0.2)] to-[rgb(var(--color-secondary)/0.2)]
            blur-sm animate-pulse
          `}
                />
            </div>

            {text && (
                <p className="text-sm font-medium text-[rgb(var(--color-text-secondary))] animate-pulse">
                    {text}
                </p>
            )}
        </div>
    );
}
