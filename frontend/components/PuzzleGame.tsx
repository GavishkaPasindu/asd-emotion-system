'use client';

import { useState, useEffect, useCallback } from 'react';
import { RefreshCcw, Star, Check } from 'lucide-react';

interface PuzzleGameProps {
    theme?: string;
    onSolve?: () => void;
}

function shuffle<T>(arr: T[]): T[] {
    const a = [...arr];
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

// ─────────────────────────────────────────────────────────────────────────────
// SHARED WIN SCREEN
// ─────────────────────────────────────────────────────────────────────────────
function WinScreen({ moves, onRestart, message }: { moves?: number, onRestart: () => void, message?: string }) {
    return (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
            <div className="bg-yellow-400 text-yellow-900 rounded-3xl p-10 text-center shadow-[0_0_50px_rgba(250,204,21,0.5)] animate-bounce-short">
                <Star className="w-20 h-20 mx-auto mb-4 fill-yellow-600 text-yellow-600" />
                <p className="text-5xl font-black tracking-tight mb-2">Great Job! 🎉</p>
                {message && <p className="text-2xl mt-2 font-bold text-yellow-800">{message}</p>}
                {moves !== undefined && <p className="text-xl mt-2 font-semibold bg-yellow-500/20 inline-block px-4 py-2 rounded-full">Solved in {moves} moves!</p>}
                <div className="mt-8">
                    <button
                        onClick={onRestart}
                        className="px-8 py-4 bg-yellow-900 text-yellow-100 rounded-full font-black text-xl hover:bg-black transition-colors shadow-xl hover:scale-105 active:scale-95 flex items-center gap-2 mx-auto"
                    >
                        <RefreshCcw className="w-6 h-6" /> Play Again
                    </button>
                </div>
            </div>
        </div>
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. ANIMAL MATCH (Reference Grid vs Playable Grid)
// ─────────────────────────────────────────────────────────────────────────────
const ANIMAL_EMOJIS = ['🐶', '🐱', '🐭', '🐹', '🐰', '🦊', '🐻', '🐼', '🐨'];

function AnimalMatch({ onSolve }: { onSolve?: () => void }) {
    const SIZE = 9;

    interface Piece { id: number; currentPos: number; correctPos: number; emoji: string; }

    const [pieces, setPieces] = useState<Piece[]>([]);
    const [selected, setSelected] = useState<number | null>(null);
    const [solved, setSolved] = useState(false);
    const [moves, setMoves] = useState(0);

    const initPuzzle = useCallback(() => {
        const shuffledPositions = shuffle([...Array(SIZE).keys()]);
        setPieces(ANIMAL_EMOJIS.map((emoji, id) => ({
            id,
            correctPos: id,
            currentPos: shuffledPositions[id],
            emoji,
        })));
        setSelected(null);
        setSolved(false);
        setMoves(0);
    }, []);

    useEffect(() => { initPuzzle(); }, [initPuzzle]);

    const handleTileClick = (pos: number) => {
        if (solved) return;
        if (selected === null) {
            setSelected(pos);
            return;
        }
        if (selected === pos) { setSelected(null); return; }

        setPieces(prev => {
            return prev.map(p => {
                if (p.currentPos === selected) return { ...p, currentPos: pos };
                if (p.currentPos === pos) return { ...p, currentPos: selected };
                return p;
            });
        });
        setMoves(m => m + 1);
        setSelected(null);
    };

    // Check for win condition after pieces update
    useEffect(() => {
        if (pieces.length === 0 || solved) return;
        const isSolved = pieces.every(p => p.currentPos === p.correctPos);
        if (isSolved) {
            setSolved(true);
            onSolve?.();
        }
    }, [pieces, solved, onSolve]);

    const getPieceAtPos = (pos: number) => pieces.find(p => p.currentPos === pos);

    return (
        <div className="flex flex-col items-center justify-center min-h-full w-full bg-gradient-to-br from-indigo-900 to-purple-900 p-4 relative py-12">
            <div className="flex items-center gap-4 mb-4 md:mb-8">
                <h2 className="text-3xl font-black text-white drop-shadow">🐶 Animal Match!</h2>
                <span className="px-3 py-1 bg-white/20 rounded-full text-white text-sm font-mono">Moves: {moves}</span>
                <button onClick={initPuzzle} className="p-2 rounded-full bg-white/20 hover:bg-white/30 text-white transition-colors" title="Restart">
                    <RefreshCcw className="w-5 h-5" />
                </button>
            </div>

            <p className="text-white/80 mb-6 text-sm md:text-base font-semibold text-center">Tap two animals on the right to swap them until they match the left!</p>

            {solved && <WinScreen moves={moves} onRestart={initPuzzle} />}

            <div className="flex flex-col md:flex-row gap-6 md:gap-16 items-center w-full max-w-4xl justify-center">
                {/* Left Side: TARGET */}
                <div className="flex flex-col items-center">
                    <h3 className="text-white/80 font-bold mb-2 md:mb-4 uppercase tracking-wider text-xs md:text-sm">Target (Match This)</h3>
                    <div className="grid grid-cols-3 gap-1 md:gap-2 p-2 md:p-3 bg-black/20 rounded-2xl">
                        {ANIMAL_EMOJIS.map((emoji, i) => (
                            <div key={i} className="w-12 h-12 sm:w-16 sm:h-16 md:w-20 md:h-20 bg-white/10 rounded-xl flex items-center justify-center text-3xl md:text-4xl shadow-inner">
                                {emoji}
                            </div>
                        ))}
                    </div>
                </div>

                {/* Right Side: PLAYABLE */}
                <div className="flex flex-col items-center">
                    <h3 className="text-yellow-300 font-bold mb-2 md:mb-4 uppercase tracking-wider text-xs md:text-sm">Your Board (Play Here)</h3>
                    <div className="grid grid-cols-3 gap-1 md:gap-2 p-2 md:p-3 bg-white/10 rounded-2xl">
                        {[...Array(SIZE)].map((_, pos) => {
                            const piece = getPieceAtPos(pos);
                            const isSelected = selected === pos;
                            const isCorrect = piece && piece.currentPos === piece.correctPos;
                            return (
                                <button
                                    key={pos}
                                    onClick={() => handleTileClick(pos)}
                                    className={`
                                        w-12 h-12 sm:w-16 sm:h-16 md:w-20 md:h-20 rounded-xl text-3xl md:text-4xl flex items-center justify-center
                                        transition-all duration-200 shadow-md font-emoji select-none
                                        ${isSelected ? 'scale-110 ring-4 ring-yellow-400 bg-white/40 z-10' :
                                            isCorrect ? 'bg-green-500/50' : 'bg-white/20 hover:bg-white/30'
                                        }
                                        active:scale-95
                                    `}
                                >
                                    {piece?.emoji ?? ''}
                                </button>
                            );
                        })}
                    </div>
                </div>
            </div>
        </div>
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. FIND THE FACE
// ─────────────────────────────────────────────────────────────────────────────
const FACES = [
    { emoji: '😊', name: 'Happy', color: 'bg-yellow-400' },
    { emoji: '😢', name: 'Sad', color: 'bg-blue-400' },
    { emoji: '😡', name: 'Angry', color: 'bg-red-400' },
    { emoji: '😮', name: 'Surprised', color: 'bg-purple-400' },
    { emoji: '😨', name: 'Scared', color: 'bg-orange-400' },
];

function FindTheFace({ onSolve }: { onSolve?: () => void }) {
    const [target, setTarget] = useState(FACES[0]);
    const [options, setOptions] = useState<typeof FACES>([]);
    const [solved, setSolved] = useState(false);
    const [errors, setErrors] = useState<number[]>([]);

    const initPuzzle = useCallback(() => {
        const shuffledFaces = shuffle(FACES);
        const selectedTarget = shuffledFaces[0];
        const currentOptions = shuffle(shuffledFaces.slice(0, 4));

        setTarget(selectedTarget);
        setOptions(currentOptions);
        setSolved(false);
        setErrors([]);
    }, []);

    useEffect(() => { initPuzzle(); }, [initPuzzle]);

    const handlePick = (idx: number, face: typeof FACES[0]) => {
        if (solved) return;
        if (face.name === target.name) {
            setSolved(true);
            // Wait a second to show the success, then loop to a new face!
            setTimeout(() => {
                initPuzzle();
            }, 1500);
        } else {
            if (!errors.includes(idx)) setErrors([...errors, idx]);
        }
    };

    return (
        <div className="flex flex-col items-center justify-center min-h-full w-full bg-gradient-to-br from-blue-900 to-indigo-900 p-4 py-12 relative text-center">

            <div className="mb-8 md:mb-12">
                <h2 className="text-3xl md:text-5xl font-black text-white drop-shadow mb-2 md:mb-4">
                    Can you find the <span className="text-yellow-300 underline">{target.name}</span> face?!
                </h2>
                <p className="text-lg md:text-xl text-white/80">Tap the correct emotion!</p>
            </div>

            {solved && (
                <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm pointer-events-none">
                    <div className="bg-green-500 text-white rounded-3xl p-8 text-center shadow-2xl animate-bounce">
                        <Check className="w-20 h-20 mx-auto mb-2" />
                        <p className="text-4xl font-black">Correct! 🎉</p>
                    </div>
                </div>
            )}

            <div className="grid grid-cols-2 md:flex md:flex-wrap justify-center gap-4 md:gap-6 max-w-4xl">
                {options.map((face, idx) => {
                    const isError = errors.includes(idx);
                    return (
                        <button
                            key={idx}
                            onClick={() => handlePick(idx, face)}
                            className={`
                                w-32 h-32 md:w-48 md:h-48 text-7xl md:text-8xl flex items-center justify-center rounded-3xl
                                shadow-xl transition-all duration-300 transform
                                ${isError ? 'bg-red-500/50 scale-95 opacity-50 cursor-not-allowed' : 'bg-white hover:scale-110 hover:-translate-y-2'}
                            `}
                            disabled={isError || solved}
                        >
                            {face.emoji}
                        </button>
                    );
                })}
            </div>

            <button onClick={initPuzzle} className="mt-12 p-3 rounded-full bg-white/20 hover:bg-white/30 text-white transition-colors" title="Restart">
                <RefreshCcw className="w-6 h-6" />
            </button>
        </div>
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. FRUIT HUNT (Select 2 Fruits among 9 items)
// ─────────────────────────────────────────────────────────────────────────────
const FRUITS = ['🍎', '🍌', '🍇', '🍉', '🍓', '🍊', '🍍', '🥝', '🍒', '🍑'];
const NON_FRUITS = ['🚗', '⚽', '🎸', '📱', '👕', '🍔', '🍕', '🏠', '🚀', '✏️', '🐶', '🏀'];

function FruitHunt({ onSolve }: { onSolve?: () => void }) {
    interface Tile { id: number; emoji: string; isFruit: boolean; isSelected: boolean; }

    const [tiles, setTiles] = useState<Tile[]>([]);
    const [solved, setSolved] = useState(false);
    const [shakeIdx, setShakeIdx] = useState<number | null>(null);

    const initPuzzle = useCallback(() => {
        const selectedFruits = shuffle(FRUITS).slice(0, 2);
        const selectedNonFruits = shuffle(NON_FRUITS).slice(0, 7);

        const allItems = shuffle([
            ...selectedFruits.map(f => ({ emoji: f, isFruit: true })),
            ...selectedNonFruits.map(nf => ({ emoji: nf, isFruit: false }))
        ]);

        setTiles(allItems.map((item, id) => ({ id, emoji: item.emoji, isFruit: item.isFruit, isSelected: false })));
        setSolved(false);
    }, []);

    useEffect(() => { initPuzzle(); }, [initPuzzle]);

    const handleTileClick = (idx: number) => {
        if (solved) return;
        const tile = tiles[idx];

        if (tile.isSelected) return;

        if (!tile.isFruit) {
            // Shake effect for wrong item
            setShakeIdx(idx);
            setTimeout(() => setShakeIdx(null), 500);
            return;
        }

        // Correct pick!
        const newTiles = [...tiles];
        newTiles[idx].isSelected = true;
        setTiles(newTiles);

        const selectedFruitsCount = newTiles.filter(t => t.isFruit && t.isSelected).length;
        if (selectedFruitsCount === 2) {
            setSolved(true);
            setTimeout(() => onSolve?.(), 500); // slight delay so they see the second checkmark
        }
    };

    return (
        <div className="flex flex-col items-center justify-center min-h-full w-full bg-gradient-to-br from-pink-900 to-rose-900 p-4 py-12 relative">
            <div className="mb-6 md:mb-8 text-center px-4">
                <h2 className="text-3xl md:text-5xl font-black text-white drop-shadow mb-2">🍎 Fruit Hunt!</h2>
                <p className="text-lg md:text-2xl text-yellow-200 font-bold">Find and tap the TWO fruits hidden here!</p>
            </div>

            {solved && <WinScreen onRestart={initPuzzle} message="You found both fruits! Yummy!" />}

            <div className="grid grid-cols-3 gap-2 md:gap-4 p-4 md:p-6 bg-black/20 rounded-3xl">
                {tiles.map((tile, idx) => (
                    <button
                        key={idx}
                        onClick={() => handleTileClick(idx)}
                        className={`
                            w-20 h-20 sm:w-24 sm:h-24 md:w-32 md:h-32 rounded-2xl text-5xl md:text-6xl flex items-center justify-center
                            transition-all duration-300 shadow-xl
                            ${tile.isSelected ? 'bg-green-500 scale-105 ring-4 ring-green-300' : 'bg-white hover:bg-gray-100 hover:scale-105'}
                            ${shakeIdx === idx ? 'animate-[shake_0.5s_ease-in-out]' : ''}
                        `}
                    >
                        {tile.isSelected ? (
                            <div className="relative">
                                {tile.emoji}
                                <div className="absolute -bottom-2 -right-2 bg-green-500 rounded-full p-1 text-white border-2 border-white">
                                    <Check className="w-4 h-4 md:w-5 md:h-5 font-bold" />
                                </div>
                            </div>
                        ) : (
                            tile.emoji
                        )}
                    </button>
                ))}
            </div>

            <button onClick={initPuzzle} className="mt-8 p-3 rounded-full bg-white/20 hover:bg-white/30 text-white transition-colors" title="Restart">
                <RefreshCcw className="w-6 h-6" />
            </button>

            <style jsx>{`
                @keyframes shake {
                    0%, 100% { transform: translateX(0); }
                    25% { transform: translateX(-10px) rotate(-5deg); }
                    50% { transform: translateX(10px) rotate(5deg); }
                    75% { transform: translateX(-10px) rotate(-5deg); }
                }
            `}</style>
        </div>
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN WRAPPER
// ─────────────────────────────────────────────────────────────────────────────
export default function PuzzleGame({ theme = 'animals', onSolve }: PuzzleGameProps) {
    if (theme === 'faces' || theme === 'space') return <FindTheFace onSolve={onSolve} />;
    if (theme === 'fruits') return <FruitHunt onSolve={onSolve} />;

    // default to animals
    return <AnimalMatch onSolve={onSolve} />;
}
