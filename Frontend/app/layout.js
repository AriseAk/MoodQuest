import "./globals.css";
import Navbar from "@/components/Navbar";
import { Luckiest_Guy } from 'next/font/google';
import { getServerSession } from 'next-auth';
import SessionProvider from '@/components/SessionProvider';

const luckiestGuy = Luckiest_Guy({
  weight: '400',
  subsets: ['latin'],
});

export const metadata = {
  title: "MoodQuest",
  description: "Your mental wellness journey",
};

export default async function RootLayout({ children }) {
  const session = await getServerSession();
  
  return (
    <html lang="en">
      <body className={`${luckiestGuy.className} antialiased`}>
        <SessionProvider session={session}>
          {children}
        </SessionProvider>
      </body>
    </html>
  );
}