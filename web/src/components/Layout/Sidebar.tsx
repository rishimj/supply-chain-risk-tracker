import React, { Fragment } from "react";
import { Link } from "react-router-dom";
import { Dialog, Transition } from "@headlessui/react";
import {
  HomeIcon,
  BuildingOfficeIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  BeakerIcon,
  Cog6ToothIcon,
  HeartIcon,
  DocumentCheckIcon,
  XMarkIcon,
  ShieldCheckIcon,
} from "@heroicons/react/24/outline";
import { clsx } from "clsx";
import type { NavigationItem } from "../../types";

interface SidebarProps {
  open: boolean;
  onClose: () => void;
  currentPath: string;
}

const navigation: NavigationItem[] = [
  { name: "Dashboard", href: "/dashboard", icon: HomeIcon },
  { name: "Companies", href: "/companies", icon: BuildingOfficeIcon },
  { name: "Predictions", href: "/predictions", icon: ExclamationTriangleIcon },
  { name: "Analytics", href: "/analytics", icon: ChartBarIcon },
];

const systemNavigation: NavigationItem[] = [
  { name: "System Health", href: "/system/health", icon: HeartIcon },
  {
    name: "Data Quality",
    href: "/system/data-quality",
    icon: DocumentCheckIcon,
  },
  { name: "Models", href: "/system/models", icon: BeakerIcon },
];

const settingsNavigation: NavigationItem[] = [
  { name: "Settings", href: "/settings", icon: Cog6ToothIcon },
];

function SidebarContent({ currentPath }: { currentPath: string }) {
  return (
    <div className="flex flex-col flex-grow bg-white overflow-y-auto border-r border-gray-200">
      {/* Logo */}
      <div className="flex items-center flex-shrink-0 px-4 py-6">
        <div className="flex items-center">
          <ShieldCheckIcon className="h-8 w-8 text-primary-600" />
          <div className="ml-3">
            <h1 className="text-xl font-bold text-gray-900">Supply Chain</h1>
            <p className="text-sm text-gray-500">Risk Tracker</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 pb-4 space-y-8">
        {/* Main Navigation */}
        <div>
          <h3 className="px-3 text-xs font-semibold text-gray-500 uppercase tracking-wider">
            Main
          </h3>
          <div className="mt-2 space-y-1">
            {navigation.map((item) => {
              const isActive =
                currentPath === item.href ||
                (item.href !== "/dashboard" &&
                  currentPath.startsWith(item.href));

              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={clsx(
                    isActive ? "sidebar-link-active" : "sidebar-link-inactive"
                  )}
                >
                  <item.icon className="mr-3 h-5 w-5" />
                  {item.name}
                  {item.badge && (
                    <span className="ml-auto bg-primary-100 text-primary-600 text-xs rounded-full px-2 py-1">
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>
        </div>

        {/* System Navigation */}
        <div>
          <h3 className="px-3 text-xs font-semibold text-gray-500 uppercase tracking-wider">
            System
          </h3>
          <div className="mt-2 space-y-1">
            {systemNavigation.map((item) => {
              const isActive = currentPath === item.href;

              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={clsx(
                    isActive ? "sidebar-link-active" : "sidebar-link-inactive"
                  )}
                >
                  <item.icon className="mr-3 h-5 w-5" />
                  {item.name}
                  {item.badge && (
                    <span className="ml-auto bg-danger-100 text-danger-600 text-xs rounded-full px-2 py-1">
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>
        </div>

        {/* Settings Navigation */}
        <div>
          <div className="mt-2 space-y-1">
            {settingsNavigation.map((item) => {
              const isActive = currentPath === item.href;

              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={clsx(
                    isActive ? "sidebar-link-active" : "sidebar-link-inactive"
                  )}
                >
                  <item.icon className="mr-3 h-5 w-5" />
                  {item.name}
                </Link>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Footer */}
      <div className="flex-shrink-0 flex border-t border-gray-200 p-4">
        <div className="flex-shrink-0 w-full group block">
          <div className="flex items-center">
            <div>
              <div className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-success-100 text-success-800">
                <div className="w-2 h-2 bg-success-400 rounded-full mr-1.5"></div>
                System Online
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function Sidebar({ open, onClose, currentPath }: SidebarProps) {
  return (
    <>
      {/* Mobile sidebar */}
      <Transition.Root show={open} as={Fragment}>
        <Dialog as="div" className="relative z-40 md:hidden" onClose={onClose}>
          <Transition.Child
            as={Fragment}
            enter="transition-opacity ease-linear duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="transition-opacity ease-linear duration-300"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <div className="fixed inset-0 bg-gray-600 bg-opacity-75" />
          </Transition.Child>

          <div className="fixed inset-0 flex z-40">
            <Transition.Child
              as={Fragment}
              enter="transition ease-in-out duration-300 transform"
              enterFrom="-translate-x-full"
              enterTo="translate-x-0"
              leave="transition ease-in-out duration-300 transform"
              leaveFrom="translate-x-0"
              leaveTo="-translate-x-full"
            >
              <Dialog.Panel className="relative flex-1 flex flex-col max-w-xs w-full">
                <Transition.Child
                  as={Fragment}
                  enter="ease-in-out duration-300"
                  enterFrom="opacity-0"
                  enterTo="opacity-100"
                  leave="ease-in-out duration-300"
                  leaveFrom="opacity-100"
                  leaveTo="opacity-0"
                >
                  <div className="absolute top-0 right-0 -mr-12 pt-2">
                    <button
                      type="button"
                      className="ml-1 flex items-center justify-center h-10 w-10 rounded-full focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
                      onClick={onClose}
                    >
                      <span className="sr-only">Close sidebar</span>
                      <XMarkIcon
                        className="h-6 w-6 text-white"
                        aria-hidden="true"
                      />
                    </button>
                  </div>
                </Transition.Child>
                <SidebarContent currentPath={currentPath} />
              </Dialog.Panel>
            </Transition.Child>
            <div className="flex-shrink-0 w-14">
              {/* Force sidebar to shrink to fit close icon */}
            </div>
          </div>
        </Dialog>
      </Transition.Root>

      {/* Static sidebar for desktop */}
      <div className="hidden md:flex md:w-64 md:flex-col md:fixed md:inset-y-0">
        <SidebarContent currentPath={currentPath} />
      </div>
    </>
  );
}
