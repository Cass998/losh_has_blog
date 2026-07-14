export function postStats(markdown: string) {
	const text = markdown.replace(/```[\s\S]*?```/g, ' ').replace(/!?(?:\[[^\]]*\]\([^)]*\))/g, ' ').replace(/[#>*_`~]/g, ' ');
	const cjk = (text.match(/[\u3400-\u9fff]/g) ?? []).length;
	const words = (text.replace(/[\u3400-\u9fff]/g, ' ').match(/[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*/g) ?? []).length;
	const wordCount = Math.max(cjk + words, 1);
	return { wordCount, readingMinutes: Math.max(1, Math.ceil(wordCount / 400)) };
}
