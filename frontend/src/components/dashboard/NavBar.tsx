const NavBar = () => {
  return (
    <header className="h-14 border-b bg-card flex items-center px-6 shrink-0">
      <div className="flex items-center gap-4">
        <div className="h-3 w-3 rounded-full bg-nominal" />
        <div>
          <h1 className="text-sm font-semibold font-mono tracking-wide text-foreground leading-none">
            ENGINESENTINEL
          </h1>
          <p className="text-[10px] font-mono text-muted-foreground tracking-widest uppercase mt-0.5">
            Intelligent Aircraft Engine Health Monitoring
          </p>
        </div>
      </div>
      <div className="ml-auto flex items-center gap-6">
        <span className="label-text">SYSTEM ACTIVE</span>
        <div className="h-2 w-2 rounded-full bg-nominal animate-pulse" />
      </div>
    </header>
  );
};

export default NavBar;
