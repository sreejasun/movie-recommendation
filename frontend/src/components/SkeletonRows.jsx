export default function SkeletonRows({ rows = 5 }) {
  return (
    <div className="space-y-3 p-6">
      {Array.from({ length: rows }).map((_, index) => (
        <div key={index} className="skeleton h-11 w-full" />
      ))}
    </div>
  );
}
