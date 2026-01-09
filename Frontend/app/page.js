import Navbar from "@/components/Navbar";
import Image from "next/image";

export default function Home() {
  return (
    <>
    <Navbar/>
    <section className="flex justify-around items-center py-50 max-h-[90vh]">
      <div className="relative h-120 w-200 ">
  <Image
    src="/girl.jpeg"
    alt="image"
    fill
    className="object-cover rounded-lg"
  />
</div>
      <div className="flex flex-col gap-8">
  <div>
    <p className="text-6xl md:text-5xl font-black text-black leading-tight">YOUR FACE</p>
    <p className="text-6xl md:text-7xl font-black text-black leading-tight">SPEAKS.</p>
    <p className="text-6xl md:text-7xl font-black text-black leading-tight">WE LISTEN!</p>
  </div>
  <div>
    <p className="text-4xl md:text-5xl font-black text-[#8BA853] leading-tight">SIGN UP NOW!</p>
    <p className="text-4xl md:text-5xl font-black text-[#8BA853] leading-tight">FOR EXCITING INSIGHTS</p>
  </div>
</div>
    </section>
    </>
  );
}
